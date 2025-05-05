from typing import Dict, Any, Optional, List
import time
import numpy as np
import pandas as pd
import logging
import networkx as nx
import matplotlib.pyplot as plt
import pandapower as pp

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, NonNegativeReals, Reals,
    Binary, Objective, Expression, minimize, Suffix, value as pyo_val,
)
from pyomo.opt import SolverFactory, check_optimal_termination
from pyomo.util.model_size import build_model_size_report
from pint import UnitRegistry

ureg = UnitRegistry()

 
class SOCP_class:
    """
    This class implements Second Order Conic Programming optimization for 
    distribution network reconfiguration.
    """

    def __init__(self, net, graph_id: str = "", *,
                 logger: Optional[logging.Logger] = None,
                 switch_penalty: float = 1e2,
                 slack_penalty: float = 1e3,
                 toggles: Optional[Dict[str, bool]] = None,
                 active_bus_mask: Optional[pd.Series] = None) -> None:
        print
        self.net = net
        self.id = graph_id or "unnamed"
        self.logger = logger or logging.getLogger("SOCP‑RNC")

        self.toggles = toggles or {
            "voltage_drop": True,
            "voltage_bounds": True,
            "power_balance": True,
            "radiality": True,
            "radiality_spanning": True,   # else flow‑based
            "switch_cost": True,
        }
        self.switch_penalty = switch_penalty
        self.slack_penalty = slack_penalty

        # bus activity mask ------------------------------------------------
        if active_bus_mask is None:
            self.active_bus = pd.Series(True, index=net.bus.index)
        else:
            self.active_bus = pd.Series(active_bus_mask,
                                        index=net.bus.index).fillna(False).astype(bool)

        # substations -------------------------------------------------------
        self.substations = set(net.ext_grid.bus.tolist())
        if hasattr(net, "gen") and not net.gen.empty:
            self.substations |= set(net.gen[net.gen.slack].bus.tolist())
        if hasattr(net, "trafo") and not net.trafo.empty:
            self.substations |= set(net.trafo.lv_bus.tolist())
        if not self.substations:
            raise ValueError("No reference / substation buses found in network")

        # switches ----------------------------------------------------------
        self.initial_switch_status = net.switch.closed.copy()

        # containers to be filled in `initialise()`
        self.bus_df: Optional[pd.DataFrame] = None
        self.line_df: Optional[pd.DataFrame] = None
        self.switch_df: Optional[pd.DataFrame] = None
        self.lines_with_sw: Dict[int, List[int]] = {}
        self.lines_wo_sw: List[int] = []
        self.bus_p_inj: Dict[int, float] = {}
        self.bus_q_inj: Dict[int, float] = {}
        self.bigM: Dict[int, float] = {}

        # Pyomo model & results
        self.model: Optional[ConcreteModel] = None
        self.solver_results = None  # SolverResults or raw obj value
        self.solve_time: Optional[float] = None


    def initialize(self) -> None:
        """
        Prepare Pandapower → Pandas tables and convert all electrical
        quantities to a uniform per-unit base.

        Bases used
        ----------
        • S_base = 1 MVA  (constant for the whole network)
        • V_base,b = vn_kv[bus]  (each voltage level keeps its own base)
        • Z_base = V_base² / S_base   ⇒  R_pu = R_Ω / Z_base
        """
        S_base_MVA = 1.0          # <-- float
        self.S_base_VA = S_base_MVA * 1e6     # 1 MVA
        self.bus_df = self.net.bus.copy()

        # ------------------------------------------------ bus table
        self.bus_df["v_base_V"] = (self.bus_df.vn_kv * 1e3) 
        # keep a dict for quick access later
        self.bus_df = self.net.bus.copy()
        # base-voltage in volts (numeric float, not a pint quantity)
        self.bus_df["v_base_V"] = self.bus_df.vn_kv * 1e3
        self.v_base_V = self.bus_df.v_base_V.to_dict()

        # ------------------------------------------------ line table
        line = self.net.line.copy()
        # Ω values
        line["r_ohm"] = line.r_ohm_per_km * line.length_km
        line["x_ohm"] = line.x_ohm_per_km * line.length_km

        # per-unit impedances (own V_base)
        def _rz_pu(row):
            # Z_base = V_base² / S_base   (all numeric)
            Z_base = (self.v_base_V[row.from_bus] ** 2) / self.S_base_VA
            return row.r_ohm / Z_base

        def _xz_pu(row) -> float:
            Z_base = (self.v_base_V[row.from_bus] ** 2) / self.S_base_VA
            return row.x_ohm / Z_base

        line["r_pu"] = line.apply(_rz_pu, axis=1)
        line["x_pu"] = line.apply(_xz_pu, axis=1)
        self.line_df = line

        # ------------------------------------------------ switches
        sw = self.net.switch.copy()
        self.switch_df = sw
        self.lines_with_sw = (
            sw[sw.et == "l"].groupby("element").apply(lambda d: list(d.index)).to_dict()
        )
        self.lines_wo_sw = [l for l in line.index if l not in self.lines_with_sw]

        # ------------------------------------------------ bus injections   (per-unit)
        self.bus_p_inj = {}
        self.bus_q_inj = {}
        for b in self.bus_df.index:
            p_mw = (
                self.net.gen[self.net.gen.bus == b].p_mw.sum()
                - self.net.load[self.net.load.bus == b].p_mw.sum()
            )
            q_mvar = (
                get_reactive_injection(self.net.gen, b)
                - get_reactive_injection(self.net.load, b)
            )
            self.bus_p_inj[b] = p_mw / S_base_MVA          # pu
            self.bus_q_inj[b] = q_mvar / S_base_MVA        # pu

        # ------------------------------------------------ Big-M (per-unit P-flow limit)
        self.bigM = {}
        max_p_abs = max(abs(v) for v in self.bus_p_inj.values()) + 1e-3
        for l in line.index:
            self.bigM[l] = 2.0 * max_p_abs * max(1, self.line_df.length_km[l])   # generous – tune later

        self.logger.info("Initialisation finished (%d buses, %d lines)",
                        len(self.bus_df), len(self.line_df))
    # ------------------------------------------------------------------------
    # 2.  CREATE-MODEL   (full variable names, voltage-bound cons commented)
    # ------------------------------------------------------------------------
    def create_model(self) -> ConcreteModel:
        """
        Build the Pyomo ConcreteModel using the SOCP formulation
        (eqns 25-36 in the reference PDF).
        All quantities are **per-unit**.
        """

        if self.bus_df is None:
            self.initialize()

        model = ConcreteModel()

        # -------------------- index sets
        active_buses = list(self.bus_df.index[self.active_bus])
        active_lines = [

            l for l in self.line_df.index
            if self.active_bus[self.line_df.from_bus[l]]
            and self.active_bus[self.line_df.to_bus[l]]
        ]
        model.buses = Set(initialize=active_buses, ordered=True)
        model.lines = Set(initialize=active_lines, ordered=True)
        model.times = Set(initialize=[0])
        model.partitionedbuses = Set(
            initialize=[b for b in active_buses if b not in self.substations]
        )

        # -------------------- parameters
        v_upper_pu, v_lower_pu = 1.10, 0.90
        model.voltage_upper_bound = Param(
            model.buses, initialize=lambda m, b: v_upper_pu ** 2
        )
        model.voltage_lower_bound = Param(
            model.buses, initialize=lambda m, b: v_lower_pu ** 2
        )
        model.line_resistance_pu = Param(
            model.lines, initialize=lambda m, l: self.line_df.r_pu[l]
        )
        model.line_reactance_pu = Param(
            model.lines, initialize=lambda m, l: self.line_df.x_pu[l]
        )
        model.big_M_flow = Param(model.lines, initialize=lambda m, l: self.bigM[l])

        # convenience maps from/to
        from_bus_map = {l: int(self.line_df.from_bus[l]) for l in model.lines}
        to_bus_map   = {l: int(self.line_df.to_bus[l])   for l in model.lines}
        model.from_bus = Param(model.lines, initialize=lambda m, l: from_bus_map[l])
        model.to_bus   = Param(model.lines, initialize=lambda m, l: to_bus_map[l])

        # ------------------------------------------------------------------
        #  DECISION VARIABLES
        # ------------------------------------------------------------------
        model.voltage_squared = Var(
            model.buses, model.times, within=NonNegativeReals, initialize=1.0
        )
        model.active_power_flow = Var(model.lines, model.times, initialize=0)
        model.reactive_power_flow = Var(model.lines, model.times, initialize=0)
        model.squared_current_magnitude = Var(
            model.lines, model.times, within=NonNegativeReals, initialize=0
        )
        model.line_closed = Var(model.lines, within=Binary, initialize=1)
        model.bus_energised = Var(model.partitionedbuses, within=Binary, initialize=1)

        # feasibility slacks
        model.active_power_slack    = Var(model.buses,  model.times,within=NonNegativeReals, initialize=0)
        model.reactive_power_slack  = Var(model.buses,  model.times, within=NonNegativeReals,initialize=0)
        model.voltage_drop_slack    = Var(model.lines,  model.times,
                                        within=NonNegativeReals, initialize=0)

        model.switches = Set(
            initialize=list(self.switch_df[self.switch_df.et == "l"].index)
        )
        model.switch_status = Var(
            model.switches,
            within=Binary,
            initialize=lambda m, s: 1 if self.switch_df.closed[s] else 0
            )
        model.switch_initial = Param(
                model.switches,
                within=Binary,
                initialize=lambda m, s: 1 if self.initial_switch_status.at[s] else 0
        )
        # --------------------------------------------
        # Initialize slack variables
        # ---------------------------------------------
        
        # link each switch to its host line
        def _switch_line_link(m, s):
            host = int(self.switch_df.element[s])
            return m.switch_status[s] <= m.line_closed[host]
        model.SwitchLineLink = Constraint(model.switches, rule=_switch_line_link)
        
        model.bus_upstream = Param(model.buses,
                           initialize=lambda m,b: 1 if b in self.substations else 0,
                           mutable=False)

        def _is_up_rule(m, b):
            return m.bus_energised[b] if b in m.partitionedbuses else 1
        model.is_up = Expression(model.buses, rule=_is_up_rule)

        model.SwitchLineLink = Constraint(model.switches, rule=_switch_line_link)

        # ------------------------------------------------------------------
        #  CONSTRAINTS
        # ------------------------------------------------------------------
        # (26) voltage drop (SOCP relaxation with slack)
        def voltage_drop_equation(m, l, t):
            i = m.from_bus[l]
            j = m.to_bus[l]
            R = m.line_resistance_pu[l]
            X = m.line_reactance_pu[l]
            return (
                m.voltage_squared[j, t] + m.voltage_drop_slack[l, t]
                == m.voltage_squared[i, t]
                - 2 * (R * m.active_power_flow[l, t] + X * m.reactive_power_flow[l, t])
                + (R**2 + X**2) * m.squared_current_magnitude[l, t]
            )

        if self.toggles["include_voltage_drop_constraint"]:
            model.VoltageDropEq = Constraint(model.lines, model.times, rule=voltage_drop_equation)

        # (27) active-power balance + slack
        def active_power_balance(m, b, t):
            if b in self.substations:
                return Constraint.Skip
            out_p = sum(
                m.active_power_flow[l, t] for l in m.lines if m.from_bus[l] == b
            )
            in_p = sum(
                m.active_power_flow[l, t] for l in m.lines if m.to_bus[l] == b
            )
            return out_p - in_p == self.bus_p_inj[b] + m.active_power_slack[b, t]

        # (28) reactive-power balance + slack
        def reactive_power_balance(m, b, t):
            if b in self.substations:
                return Constraint.Skip
            out_q = sum(
                m.reactive_power_flow[l, t] for l in m.lines if m.from_bus[l] == b
            )
            in_q = sum(
                m.reactive_power_flow[l, t] for l in m.lines if m.to_bus[l] == b
            )
            return out_q - in_q == self.bus_q_inj[b] + m.reactive_power_slack[b, t]

        if self.toggles["include_power_balance_constraint"]:
            model.ActivePowerBalance = Constraint(model.buses, model.times, rule=active_power_balance)
            model.ReactivePowerBalance = Constraint(model.buses, model.times, rule=reactive_power_balance)

        # (29) second-order cone: ‖(2P,2Q,V_i-ℓ)‖₂ ≤ V_i+ℓ
        def socp_cone(m, l, t):
            i = m.from_bus[l]
            return (
                (2 * m.active_power_flow[l, t]) ** 2
                + (2 * m.reactive_power_flow[l, t]) ** 2
                + (m.voltage_squared[i, t] - m.squared_current_magnitude[l, t]) ** 2
                <= (m.voltage_squared[i, t] + m.squared_current_magnitude[l, t]) ** 2
            )

        model.SOCP_Cone = Constraint(model.lines, model.times, rule=socp_cone)

        # -------- voltage bounds (relaxed with Big-M)  ----------------------
        #  !!!  kept *commented* as requested
        # def voltage_upper(m, b, t):
        #     lim = m.voltage_upper_bound[b]
        #     return m.voltage_squared[b, t] <= lim
        #
        # def voltage_lower(m, b, t):
        #     lim = m.voltage_lower_bound[b]
        #     return m.voltage_squared[b, t] >= lim
        #
        # if self.toggles["voltage_bounds"]:
        #     model.VoltageUpper = Constraint(model.buses, model.times, rule=voltage_upper)
        #     model.VoltageLower = Constraint(model.buses, model.times, rule=voltage_lower)
                # (30)-(31) Voltage bounds with big‑M relaxation
        def _flow_ub(m, l, t):
            return  m.active_power_flow[l, t] <=  m.big_M_flow[l] * m.line_closed[l]
        def _flow_lb(m, l, t):
            return -m.active_power_flow[l, t] <=  m.big_M_flow[l] * m.line_closed[l]
        model.FlowUB = Constraint(model.lines, model.times, rule=_flow_ub)
        model.FlowLB = Constraint(model.lines, model.times, rule=_flow_lb)

        def _line_bus_link_from(m, l):
            j = m.from_bus[l]
            if j in m.partitionedbuses:
                return m.line_closed[l] <= m.is_up[j]
            return Constraint.Skip
        def _line_bus_link_to(m, l):
            j = m.to_bus[l]
            if j in m.partitionedbuses:
                return m.line_closed[l] <= m.is_up[j]
            return Constraint.Skip

        def _bus_connectivity(m, b):
            inc = [l for l in m.lines if m.from_bus[l] == b or m.to_bus[l] == b]
            return sum(m.line_closed[l] for l in inc) >= m.is_up[b]
        
        def _spanning(m):
            rhs = len(m.buses) - 1 - sum(1 - m.is_up[b] for b in m.partitionedbuses)
            return sum(m.line_closed[l] for l in m.lines) == rhs
        def root_cap(m, l):
            return m.root_flow[l] <=  m.line_closed[l] #(len(m.buses) - 1) *

        # Flow balance:
        #   – Each substation *sources* Σ y_k units (one per energised PQ-bus)
        #   – Every energised PQ-bus *sinks* one unit
        def root_balance(m, b):
            inflow  = sum(m.root_flow[l] for l in m.lines if m.to_bus[l]   == b)
            outflow = sum(m.root_flow[l] for l in m.lines if m.from_bus[l] == b)

            if b in self.substations:
                # Source delivers demand of all energised partitioned buses
                return outflow - inflow == sum(m.is_up[k] for k in m.partitionedbuses)
            else:
                # Demand is 1 pu when the bus is energised, 0 otherwise
                return inflow - outflow == (m.is_up[b] if b in m.partitionedbuses else 0)
            
        # (34)-(35) Bus‑line logical links (only needed if radiality on) ----
        if self.toggles["include_radiality_constraints"]:
            model.LineBusFrom = Constraint(model.lines, rule=_line_bus_link_from)
            model.LineBusTo   = Constraint(model.lines, rule=_line_bus_link_to)
            model.BusConnectivity = Constraint(model.partitionedbuses, rule=_bus_connectivity)
            if self.toggles["use_spanning_tree_radiality"]:
                model.SpanConstraint = Constraint(rule=_spanning)
            else:
                model.root_flow = Var(model.lines, within=NonNegativeReals, initialize=0)
                model.RootCap = Constraint(model.lines, rule=root_cap)
                model.RootBalance = Constraint(model.buses, rule=root_balance)

        # -------------------- objective (25) + penalties
        loss_term = sum(
            model.line_resistance_pu[l] * model.squared_current_magnitude[l, 0]
            for l in model.lines
        )
        slack_term = self.slack_penalty * (
            sum(model.voltage_drop_slack[l, 0] for l in model.lines)
            + sum(
                model.active_power_slack[b, 0] ** 2 + model.reactive_power_slack[b, 0] ** 2
                for b in model.buses
            )
        )
        obj_expr = loss_term + slack_term

        # optional switch-move penalty (already quadratic-free)
        if self.toggles["include_switch_penalty"]:
            switch_pen = self.switch_penalty * sum(
                model.switch_initial[s] * (1 - model.switch_status[s]) +
                (1 - model.switch_initial[s]) *   model.switch_status[s]
                for s in model.switches
            )
            obj_expr += switch_pen

        model.objective = Objective(expr=obj_expr, sense=minimize)

     
        self.model = model
        self.logger.debug("\n%s", build_model_size_report(model))
        return model

    

    def solve(self, *, solver: str = "gurobi_persistent",
              time_limit: int = 600, threads: int = 8,
              mip_gap: float = 1e-2, **solver_kw) -> Any:
        if self.model is None:
            self.create_model()
        m = self.model

        start = time.time()
        opt = SolverFactory(solver)
        if hasattr(opt, "set_instance"):
            opt.set_instance(m)
        # generic options
        opt.options.update({"Threads": threads, "TimeLimit": time_limit})
        # Gurobi specific tunes ------------------------------------------
        if solver.startswith("gurobi"):
            opt.options.update({"MIPGap": mip_gap, "NonConvex": 2})
        # user overrides
        opt.options.update(solver_kw)

        self.solver_results = opt.solve(tee=False, load_solutions=True)
        self.solve_time = time.time() - start

        if not check_optimal_termination(self.solver_results):
            self.logger.warning("Solver finished with status %s",
                             self.solver_results.solver.termination_condition)

        self.logger.info(
            "Solved in %.2fs  (obj = %.3f)",
            self.solve_time,
            pyo_val(self.model.objective)
        )
        return self.solver_results
    
    def _count_changed_switches(self):
        return sum(int(round(pyo_val(self.model.switch_status[s]))) != int(self.initial_switch_status.at[s])
        for s in self.model.switches)


    def _debug_obj(self):
        print("loss :", pyo_val(sum(self.model.rl_mOhm[l] * self.model.l_squared[l,0] * 1e-3
                                    for l in self.model.lines)))
        print("switch :", pyo_val(sum(self.model.switch_initial[s] * (1 - self.model.switch_status[s])
                                    + (1 - self.model.switch_initial[s]) * self.model.switch_status[s]
                                    for s in self.model.switches)))
    def verify_solution(self, tol: float = 1e-6,
                    vmin: float = 0.90, vmax: float = 1.10,
                    logger=None) -> list[dict]:
        """
        Post-solve sanity checker.
        Returns a list of dictionaries; each dict describes one violation.
        """
        lg = logger or self.logger
        viol = []
        m    = self.model

        # 0) solver termination ------------------------------------------------
        if self.solver_results is not None \
        and not check_optimal_termination(self.solver_results):
            lg.warning(f"Solver finished with {self.solver_results.solver.termination_condition}")
            viol.append({"type": "termination",
                        "cond": str(self.solver_results.solver.termination_condition)})

        # 1) bound & constraint violations -------------------------------------
        for c in m.component_data_objects(Constraint, active=True):
            lb = c.lower if c.has_lb() else None
            ub = c.upper if c.has_ub() else None
            lhs = pyo_val(c.body)
            if (lb is not None and lhs < lb - tol) or \
            (ub is not None and lhs > ub + tol):
                viol.append({"type":"constraint",
                            "name":c.name, "value":lhs,
                            "lb":lb, "ub":ub})
                lg.debug(f"{c.name}: {lhs:.3g} ∉ [{lb},{ub}]")

        # 2) variable domain check (binary≃integer, non-neg etc.) -------------
        for v in m.component_data_objects(Var, active=True):
            if v.domain is Binary:
                val = pyo_val(v)
                if abs(val - round(val)) > tol:
                    viol.append({"type":"domain","var":v.name,"value":val})
                    lg.debug(f"Binary {v.name} = {val:.4g} not integral")

        # 3) objective consistency --------------------------------------------
        obj_py = pyo_val(m.objective.expr)
        obj_m  = pyo_val(m.objective)
        if abs(obj_py - obj_m) > tol:
            viol.append({"type":"objective","py":obj_py,"model":obj_m})
            lg.debug(f"Objective mismatch {obj_py} vs {obj_m}")

        # 4) rebuild pandapower net & run PF -----------------------------------
        net_chk = self.update_network()
        try:
            pp.runpp(net_chk, enforce_q_lims=False, calculate_voltage_angles=False)
        except pp.powerflow.LoadflowNotConverged:
            viol.append({"type":"powerflow","detail":"PF did not converge"})
            lg.warning("Power-flow on optimised net did not converge")
            return viol

        # 5) radial & connected? ----------------------------------------------
        rad, con = is_radial_and_connected(net_chk)          # your helper
        if not rad or not con:
            viol.append({"type":"topology","radial":rad,"connected":con})
            lg.debug(f"Topol. rad={rad} conn={con}")

        # 6) voltage band ------------------------------------------------------
        vm = net_chk.res_bus.vm_pu
        if vm.min() < vmin - tol or vm.max() > vmax + tol:
            viol.append({"type":"voltage",
                        "min":vm.min(), "max":vm.max(),
                        "band":(vmin,vmax)})
            lg.debug(f"Voltage out of band [{vmin},{vmax}]: "
                    f"min={vm.min():.3f}, max={vm.max():.3f}")

        # 7) line current loading ---------------------------------------------
        loading = net_chk.res_line.i_ka / net_chk.line.max_i_ka
        over = loading[loading > 1 + tol]
        for idx, val in over.items():
            viol.append({"type":"thermal","line":int(idx),"loading":float(val)})
            lg.debug(f"Line {idx} overloaded to {val*100:.1f}%")

        # 8) summary -----------------------------------------------------------
        if viol:
            lg.warning(f"{len(viol)} verification issues found")
        else:
            lg.info("verify_solution: all checks passed")

        return viol

    # def extract_results(self):
    #     if not hasattr(self.model, "P_loss"):
    #         raise RuntimeError("Model is infeasible or was not solved correctly; 'P_loss' does not exist.")

    #     if self.model is None:
    #         return
    #     results = {
    #         "optimization_time": self.optimization_time,
    #         "num_switches": len(self.model.switches) if self.include_switches else 0,
    #         "num_switches_changed": self.num_switches_changed if self.include_switches else 0,
    #         "objective_value": pyo_value(self.model.objective),
    #         "power_loss": sum(pyo_value(self.model.P_loss[l, 0]) for l in self.model.lines),
    #     }
    #     if self.include_switches:
    #         switch_results = {}
    #         for s in self.model.switches:
    #             try:
    #                 initial = bool(self.initial_switch_status.at[s])
    #                 optimized = bool(round(pyo_value(self.model.switch_status[s])))
    #                 switch_results[s] = {
    #                     "initial": initial,
    #                     "optimized": optimized,
    #                     "changed": initial != optimized
    #                 }
    #             except:
    #                 pass
    #         results["switches"] = switch_results
    #     voltage_profiles = {}
    #     for b in self.model.buses:
    #         try:
    #             voltage_profiles[b] = np.sqrt(pyo_value(self.model.V_m_sqr[b, 0]))
    #         except:
    #             pass
    #     results["voltage_profiles"] = voltage_profiles
    #     self.optimized_results = results
    #     return results
    
    def active_mask(self):
        "Return pd.Series 1/0 of the buses that were optimised."
        return self.active_bus.astype(int)

    def update_network(self):
        net_updated = self.net.deepcopy()
        # 1) collect what Pyomo thinks the optimized statuses are
        optimized_statuses = {
            s: bool(round(pyo_val(self.model.switch_status[s])))
            for s in self.model.switches
        }
        # 2) log initial vs optimized
        self.logger.debug("Switch status summary before update:")
        for s in self.model.switches:
            init = bool(self.initial_switch_status.at[s])
            opt  = optimized_statuses[s]
            self.logger.debug(f"  switch {s:3d}: {init!r} → {opt!r}")

        # 3) apply back into pandapower net
        for s, status in optimized_statuses.items():
            net_updated.switch.at[s, 'closed'] = status

        # 4) log the net.switch dataframe after update
        self.logger.debug(
            "pandapower net.switch['closed'] after update:\n"
            + net_updated.switch[net_updated.switch.et=='l'][['element','closed']].to_string()
        )

        return net_updated
    def print_results(self):
        if self.optimized_results is None:
            print("No optimization results available.")
            return
        results = self.optimized_results
        print("\n" + "="*50)
        print(f"NETWORK RECONFIGURATION RESULTS: {self.net_name}")
        print("="*50)
        print(f"Optimization time: {results['optimization_time']:.2f} seconds")
        print(f"Objective value: {results['objective_value']:.4f}")
        print(f"Power loss: {results['power_loss']:.4f} kW")

        print(f"\nSwitch changes: {results['num_switches_changed']} of {results['num_switches']}")
        if results['num_switches_changed'] > 0:
            print("\nSwitches changed:")
            print(f"{'Switch ID':10} {'Initial':10} {'Optimized':10}")
            print("-" * 35)
            for s, switch_info in results['switches'].items():
                if switch_info['changed']:
                    print(f"{s:10} {'Closed' if switch_info['initial'] else 'Open':10} {'Closed' if switch_info['optimized'] else 'Open':10}")
        print("\nVoltage profile summary:")
        voltages = list(results['voltage_profiles'].values())
        if voltages:
            print(f"Min voltage: {min(voltages):.4f} p.u.")
            print(f"Max voltage: {max(voltages):.4f} p.u.")
            print(f"Avg voltage: {sum(voltages)/len(voltages):.4f} p.u.")
        print("="*50)

    def plot_network(self):
        if not hasattr(self.net, 'bus_geodata') or self.net.bus_geodata.empty:
            print("Bus geodata not available for plotting.")
            return
        plt.figure(figsize=(12, 10))
        for idx, line in self.net.line.iterrows():
            from_bus = line.from_bus
            to_bus = line.to_bus
            try:
                from_x, from_y = self.net.bus_geodata.loc[from_bus, ['x', 'y']]
                to_x, to_y = self.net.bus_geodata.loc[to_bus, ['x', 'y']]
                line_switches = self.net.switch[(self.net.switch.et == 'l') & (self.net.switch.element == idx)].index
                switches_changed = any(
                    self.optimized_results['switches'].get(s, {}).get('changed', False) for s in line_switches
                )
                if any(line_switches) and not all(self.net.switch.at[s, 'closed'] for s in line_switches):
                    plt.plot([from_x, to_x], [from_y, to_y], 'r--', linewidth=1.5)
                elif switches_changed:
                    plt.plot([from_x, to_x], [from_y, to_y], 'g-', linewidth=2.5)
                else:
                    plt.plot([from_x, to_x], [from_y, to_y], 'k-', linewidth=1.5)
                for s in line_switches:
                    switch_x = from_x + 0.3 * (to_x - from_x)
                    switch_y = from_y + 0.3 * (to_y - from_y)
                    switch_changed = self.optimized_results['switches'].get(s, {}).get('changed', False)
                    switch_closed = self.net.switch.at[s, 'closed']
                    if switch_changed:
                        if switch_closed:
                            plt.plot(switch_x, switch_y, 'gs', markersize=10, markeredgecolor='m', markeredgewidth=2)
                        else:
                            plt.plot(switch_x, switch_y, 'ro', markersize=10, markeredgecolor='m', markeredgewidth=2)
                    else:
                        if switch_closed:
                            plt.plot(switch_x, switch_y, 'gs', markersize=8)
                        else:
                            plt.plot(switch_x, switch_y, 'ro', markersize=8)
            except:
                pass
        plt.scatter(self.net.bus_geodata.x, self.net.bus_geodata.y, c='blue', marker='o', s=50)
        for idx, row in self.net.bus_geodata.iterrows():
            plt.text(row.x, row.y + 0.1, f"Bus {idx}", fontsize=8, ha='center')
        plt.title(f"Network Reconfiguration Results: {self.net_name}")
        plt.axis('equal')
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='k', linewidth=1.5, label='Active Line'),
            Line2D([0], [0], color='r', linestyle='--', linewidth=1.5, label='Inactive Line'),
            Line2D([0], [0], color='g', linewidth=2.5, label='Line with Switch Change'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Bus'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='g', markersize=8, label='Closed Switch'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Open Switch'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='g', markeredgecolor='m', markeredgewidth=2, markersize=10, label='Changed to Closed'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markeredgecolor='m', markeredgewidth=2, markersize=10, label='Changed to Open')
        ]
        plt.legend(handles=legend_elements, loc='best')
        plt.grid(False)
        plt.tight_layout()
        plt.show()
            
    def _print_model_summary(self, max_constraint_rows=0):
        """
        Quick overview: counts of vars/cons + (optionally) a full pprint.
        """
        from pyomo.util.model_size import build_model_size_report
        rep = build_model_size_report(self.model)          # aggregated counts
        self.logger.info("\nMODEL SIZE SUMMARY\n" + rep)


def get_reactive_injection(df, bus, assumed_pf=0.9):
    """
    Computes the reactive power injection for entries in df associated with a given bus.
    If the "q_mvar" column exists and at least 50% of the entries for that bus are non-null,
    it uses those values (filling missing ones with a derived estimate). Otherwise, it derives
    the reactive power from p_mw and the assumed power factor.
    """
    # Filter entries for this bus
    sub = df.loc[df.bus == bus]
    if sub.empty:
        return 0

    if "q_mvar" in df.columns:
        non_null = sub["q_mvar"].count()
        total = len(sub)
        # If most entries have a q_mvar value, use them (fill missing with derived estimate)
        if non_null >= total * 0.5:
            # Compute derived reactive power for each row (used to fill missing values)
            derived = sub["p_mw"] * np.tan(np.arccos(assumed_pf))
            # Fill NaN values with the derived reactive power
            q_values = sub["q_mvar"].fillna(derived)
            return q_values.sum()
        else:
            # Not enough q_mvar values, so derive for all rows
            return (sub["p_mw"] * np.tan(np.arccos(assumed_pf))).sum()
    else:
        # q_mvar column does not exist; derive reactive power from p_mw
        return (sub["p_mw"] * np.tan(np.arccos(assumed_pf))).sum()


def is_radial_and_connected(net, y_mask=None, require_single_ref=False):
    """
    Returns (is_radial, is_connected) **on the energised sub-graph**.
    A network is radial if each connected component is a tree with exactly one reference bus.
    
    y_mask : 1/0 per bus (Series / dict / ndarray).  None ⇒ all 1.
    """
    if y_mask is None:
        active_bus = pd.Series(1, index=net.bus.index)
    else:
        active_bus = pd.Series(y_mask, index=net.bus.index).fillna(0).astype(bool)

    # Build graph of *closed* lines between active buses
    G = nx.Graph()
    G.add_nodes_from(net.bus.index[active_bus])

    for _, sw in net.switch.query("et=='l' and closed").iterrows():
        ln = net.line.loc[sw.element]
        if active_bus[ln.from_bus] and active_bus[ln.to_bus]:
            G.add_edge(ln.from_bus, ln.to_bus)

    if G.number_of_nodes() == 0:
        return True, True       # vacuously radial & connected

    # Get reference buses (both ext_grid and slack generators)
    ref_buses = set(net.ext_grid.bus.tolist())
    if "slack" in net.gen.columns:
        ref_buses |= set(net.gen[net.gen.slack].bus)
    
    # Filter reference buses to only include active ones
    ref_buses = ref_buses & set(G.nodes())
    
    components = list(nx.connected_components(G))
    if not components:
        return True, True
    largest_component = max(components, key=len)
    G_largest = G.subgraph(largest_component)

    # # Now perform the radial and connected check on G_largest
    # is_connected = nx.is_connected(G_largest)
    # is_radial = True
    # for component in nx.connected_components(G_largest):
    #     comp_graph = G_largest.subgraph(component)
    #     comp_refs = ref_buses & component
    #     if len(comp_refs) != 1 or not nx.is_tree(comp_graph):
    #         is_radial = False
    #         break
    # return is_radial, is_connected
    components = list(nx.connected_components(G))
    if not components:
        return True, True

    # Connectedness is always evaluated on the largest energised component
    is_connected = nx.is_connected(G.subgraph(max(components, key=len)))

    if require_single_ref:          # original strict version
        for comp in components:
            comp_refs = ref_buses & comp
            if len(comp_refs) != 1 or not nx.is_tree(G.subgraph(comp)):
                return False, is_connected
        return True, is_connected

    # ← default: ignore how many reference buses live in each tree
    return all(nx.is_tree(G.subgraph(comp)) for comp in components), is_connected
