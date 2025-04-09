import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pint import UnitRegistry
from pyomo.opt import TerminationCondition
import networkx as nx

#ureg = UnitRegistry()
import logging
from pyomo.environ import (
    ConcreteModel,
    Set,
    Param,
    Var,
    Constraint,
    ConstraintList,
    NonNegativeReals,
    Binary,
    Reals,
    Objective,
    Expression,
    minimize,
    SolverFactory,
    Any,
    Suffix,
)
from pyomo.environ import value as pyo_value
from pyomo.util.infeasible import log_infeasible_constraints


class SOCP_class:
    """
    This class implements Second Order Conic Programming optimization for 
    distribution network reconfiguration.
    """

    def __init__(
        self,
        net,
        graph_id: str = "",
        switch_penalty: float = 0.01,
        logger=None,
        toggles= None,
        ):
        self.net = net
        #self.net_name = net_name
        self.logger = logger or logging.getLogger(__name__)
        self.graph_id = graph_id
        if hasattr(self.net, 'res_line') and hasattr(self.net.res_line, 'pl_mw'):
            active_power_loss = self.net.res_line.pl_mw
            self.total_active_power_loss = sum(active_power_loss)
        else:
            self.total_active_power_loss = None
        self.bus_injection = None
        self.toggles = toggles

        # Initialize dataframes to None
        self.line_df = None
        self.bus_df = None
        self.bus_dict = None
        self.voltages_bus = None
        self.B_df = None
        self.switch_df = None
        
        # Determine substations from slack generators, external grids, and transformer LV buses.
        slack_buses = set()
        if hasattr(self.net, 'gen') and not self.net.gen.empty:
            slack_buses = set(self.net.gen[self.net.gen.slack].bus.tolist())
        ext_grid_buses = set()
        if hasattr(self.net, 'ext_grid') and not self.net.ext_grid.empty:
            ext_grid_buses = set(self.net.ext_grid.bus.tolist())
        transformer_lv_buses = set()
        if hasattr(self.net, 'trafo') and not self.net.trafo.empty:
            transformer_lv_buses = set(self.net.trafo.lv_bus.tolist())
        self.substations = slack_buses.union(ext_grid_buses, transformer_lv_buses)
        if not self.substations:
            raise ValueError("No substations found in the network.")

        # Switches
        self.has_switches = hasattr(self.net, 'switch') and not self.net.switch.empty
        self.include_switches = self.has_switches
        self.switch_penalty = switch_penalty
        self.initial_switch_status = None

        # Store optimization model
        self.model = None
       
        # Performance metrics and results
        self.optimization_time = None
        self.num_switches_changed = None
        self.optimized_results = None
        self.lines_with_switches = {}

    def initialize(self):
       
        bus_df = self.net.bus.copy()
        bus_dict = {index: name for index, name in zip(self.net.bus.index, self.net.bus.name)}
        # Exclude HV sides of transformers
        if hasattr(self.net, 'trafo') and not self.net.trafo.empty:
            for index, row in self.net.trafo.iterrows():
                hv_bus = bus_dict[row["hv_bus"]]
                condition = bus_df.name == hv_bus
                if any(condition):
                    idx = bus_df[condition].index[0]
                    bus_df.drop(idx, inplace=True)
    
        self.bus_dict = {row['name']: row['name'] for idx, row in bus_df.iterrows()}
        voltages_bus = bus_df[["name", "vn_kv"]].set_index("name")
        self.voltages_bus = voltages_bus
        
        # Process line data with unit checking:
        line_df = self.net.line.copy()
        # Compute resistance in ohm 
        # line_df["r_ohm"] = line_df.apply(
        #     lambda row: (row["r_ohm_per_km"] * (ureg.ohm/ureg.km) * row["length_km"] * ureg.km).to(ureg.ohm).magnitude,
        #     axis=1
        # )
        # # Compute reactance in ohm similarly:
        # line_df["x_ohm"] = line_df.apply(
        #     lambda row: (row["x_ohm_per_km"] * (ureg.ohm/ureg.km) * row["length_km"] * ureg.km).to(ureg.ohm).magnitude,
        #     axis=1
        # )
        line_df["r_ohm"] = line_df["r_ohm_per_km"] * line_df["length_km"]
        line_df["x_ohm"] = line_df["x_ohm_per_km"] * line_df["length_km"]

        self.switch_df = self.net.switch.copy()
        self.initial_switch_status = self.switch_df['closed'].copy()
        line_switches = self.switch_df[self.switch_df['et'] == 'l']
        print(f"Initialized {len(line_switches)} line switches for optimization")
        self.lines_with_switches = {}
        for s_idx, switch in line_switches.iterrows():
            line_idx = switch.element
            if line_idx in self.net.line.index:
                line_name = self.net.line.at[line_idx, 'name']
                if line_name not in self.lines_with_switches:
                    self.lines_with_switches[line_name] = []
                self.lines_with_switches[line_name].append(s_idx)

        self.bus_injection = {
            self.net.bus.loc[idx, "name"]:
                (self.net.gen[self.net.gen.bus == idx]["p_mw"].sum() -
                self.net.load[self.net.load.bus == idx]["p_mw"].sum())
            for idx in self.net.bus.index
        }
        self.reactive_bus_injection = {}
        assumed_pf = 0.9  # Adjust as needed
        for idx in self.net.bus.index:
            bus_name = self.net.bus.loc[idx, "name"]
            q_gen = (
                get_reactive_injection(self.net.gen, idx, assumed_pf)
                if hasattr(self.net, 'gen') and not self.net.gen.empty
                else 0
            )
            q_sgen = (
                get_reactive_injection(self.net.sgen, idx, assumed_pf)
                if hasattr(self.net, 'sgen') and not self.net.sgen.empty
                else 0
            )
            q_shunt = (
                get_reactive_injection(self.net.shunt, idx, assumed_pf)
                if hasattr(self.net, 'shunt') and not self.net.shunt.empty
                else 0
            )
            q_load = (
                get_reactive_injection(self.net.load, idx, assumed_pf)
                if hasattr(self.net, 'load') and not self.net.load.empty
                else 0
            )
            # Note: For loads, ensure that your sign convention is correct.
            self.reactive_bus_injection[bus_name] = q_gen + q_sgen + q_shunt - q_load
            # Optional: Log injections for checking
            if abs(self.reactive_bus_injection[bus_name]) > 1e-6:
                 self.logger.debug(f"Bus {bus_name}: Reactive Injection = {self.reactive_bus_injection[bus_name]:.4f}")

        # Save processed data as attributes
        self.line_df = line_df
        self.bus_df = bus_df
        self.bus_dict = bus_dict
        self.voltages_bus = voltages_bus
    def initialize_with_alternative_mst(self, penalty=1.0):
        """
        Builds an alternative radial configuration by computing a Minimum Spanning Tree (MST)
        on a graph where edges currently active receive an added penalty.
        Updates switch_df and counts how many switches change.
        """
        # Build a set of current active edges using line names and mapped bus names.
        current_active_edges = set()
        for row in self.line_df.itertuples():
            line_name = row.name
            # Get the from and to bus (using your bus_dict)
            from_bus = self.bus_dict.get(row.from_bus)
            to_bus = self.bus_dict.get(row.to_bus)
            if from_bus is None or to_bus is None:
                continue
            # Determine if this line is active: if it has a switch and at least one is closed.
            is_active = False
            if line_name in self.lines_with_switches:
                for s in self.lines_with_switches[line_name]:
                    if self.initial_switch_status.at[s]:
                        is_active = True
                        break
            if is_active:
                # Store edge as a tuple (from, to) using bus names
                current_active_edges.add((from_bus, to_bus))
        
        # Create a NetworkX graph with all buses as nodes.
        G = nx.Graph()
        for bus in self.voltages_bus.index:
            G.add_node(bus)
        
        # Add each line as an edge with a weight.
        # Use the line resistance as the base weight.
        for row in self.line_df.itertuples():
            line_name = row.name
            from_bus = self.bus_dict.get(row.from_bus)
            to_bus = self.bus_dict.get(row.to_bus)
            if from_bus is None or to_bus is None:
                continue
            base_weight = row.r_ohm  # you might choose another base cost (or combine r and x)
            # If the current configuration uses this edge, add the penalty.
            if (from_bus, to_bus) in current_active_edges or (to_bus, from_bus) in current_active_edges:
                weight = base_weight + penalty
            else:
                weight = base_weight
            G.add_edge(from_bus, to_bus, weight=weight, line_name=line_name)
        
        # Compute the alternative MST
        alt_mst = nx.minimum_spanning_tree(G)
        
        # Create a set of edges (as frozensets to account for undirected order) that are in the alternative MST
        alt_edges = {frozenset((u, v)) for u, v in alt_mst.edges()}
        
        # Now update switch statuses:
        # For each line that has switches, if its endpoints (mapped via bus_dict) are in the alternative MST, set it to closed (True);
        # otherwise, open (False)
        switches_changed = 0
        for line_name, switch_list in self.lines_with_switches.items():
            # Retrieve endpoints from the line_df (assume unique line names)
            try:
                line_row = self.line_df.set_index("name").loc[line_name]
            except KeyError:
                continue
            from_bus = self.bus_dict.get(line_row["from_bus"])
            to_bus = self.bus_dict.get(line_row["to_bus"])
            if from_bus is None or to_bus is None:
                continue
            edge_key = frozenset((from_bus, to_bus))
            # Decide the new status: closed if the edge is in alt_edges, else open.
            new_status = edge_key in alt_edges
            # Update every switch associated with this line.
            for s in switch_list:
                old_status = self.initial_switch_status.at[s]
                if old_status != new_status:
                    switches_changed += 1
                # Update the switch status in your DataFrame (or store in a new attribute)
                self.switch_df.at[s, 'closed'] = new_status
        self.num_switches_changed = switches_changed
        self.logger.info(f"Alternative MST computed. {switches_changed} switches changed from the initial configuration.")


    def create_model(self, toggles=None):
        """
        Creates a SOCP (Second Order Cone Programming) model for distribution network reconfiguration
        based on the formulation provided in the PDF.
        
        The model follows the SOCP formulation from pages 3-4 of the PDF, with equations (25)-(36).
        
        Args:
            toggles: Dictionary of boolean flags to control which constraints are included

        Returns:
            model: A Pyomo ConcreteModel instance with the SOCP formulation
        """
        if toggles is None:
            toggles = {
                "include_voltage_drop_constraint": True, 
                "include_voltage_bounds_constraint": True,   
                "include_power_balance_constraint": True,  
                "include_radiality_constraints": True,
                "use_spanning_tree_radiality": True,  
                "include_switch_penalty": False
            }

        model = ConcreteModel()
        
        # Basic sets
        # Ωb: Set of all buses/nodes
        model.buses = Set(initialize=list(self.voltages_bus.index), ordered=True)
        # Ωl: Set of all lines/branches
        model.lines = Set(initialize=list(self.line_df["name"]), ordered=True)
        model.times = Set(initialize=[0], ordered=True)
        # Ωbp: Set of partitioned buses (non-substation buses) as in equations (10) and (33)
        model.partitioned_buses = Set(initialize=[b for b in model.buses if b not in self.substations], ordered=True)
        
        # Voltage parameters
        # V̄: Upper voltage limit in equation (30)
        v_upper_factor = 1.10  # Example: 1.10 pu
        v_lower_factor = 0.90  # Example: 0.90 pu

        # Ensure you are squaring the pu value for V_m_sqr bounds
        # model.V_overline = Param(
        #     model.buses,
        #     initialize=lambda m, b: ( (self.voltages_bus.at[b, 'vn_kv'] * ureg.kV * v_upper_factor).to(ureg.volt).magnitude )**2 
        # )
        # model.V_underline = Param(
        #     model.buses,
        #     initialize=lambda m, b: ( (self.voltages_bus.at[b, 'vn_kv'] * ureg.kV * v_lower_factor).to(ureg.volt).magnitude )**2
        # )

        model.V_overline = Param(
            model.buses,
            initialize=lambda m, b: (self.voltages_bus.at[b, 'vn_kv'] * 1000 * v_upper_factor)**2
        )
        model.V_underline = Param(
            model.buses,
            initialize=lambda m, b: (self.voltages_bus.at[b, 'vn_kv'] * 1000 * v_lower_factor)**2
        )
        
        # xᵢⱼ: Line reactance in equation (26)
        # model.xl_mOhm = Param(
        #     model.lines,
        #     within=NonNegativeReals,
        #     initialize=lambda m, l: ((self.line_df.set_index("name").at[l, "x_ohm"] * ureg.ohm)
        #                                 .to(ureg.mohm).magnitude)
        # )
        # # rᵢⱼ: Line resistance in equations (25) and (26)
        # model.rl_mOhm = Param(
        #     model.lines,
        #     within=NonNegativeReals,
        #     initialize=lambda m, l: ((self.line_df.set_index("name").at[l, "r_ohm"] * ureg.ohm)
        #                                 .to(ureg.mohm).magnitude)
        # )
        model.xl_mOhm = Param(
            model.lines,
            within=NonNegativeReals,
            initialize=lambda m, l: self.line_df.set_index("name").at[l, "x_ohm"] * 1e3
        )
        model.rl_mOhm = Param(
            model.lines,
            within=NonNegativeReals,
            initialize=lambda m, l: self.line_df.set_index("name").at[l, "r_ohm"] * 1e3
        )
        
        # Demand parameter for radiality constraint - used in flow balance for radiality
        model.radial_demand = Param(model.buses, initialize=lambda m, b: 0 if b in self.substations else 1)
        
        # Big M for flow constraints
        expected_max_flow = sum(abs(val) for val in self.bus_injection.values())
        print("Expected maximum flow (used for Big‑M):", expected_max_flow)
        M_value = expected_max_flow * 1.1  # Safety factor
        model.big_M = Param(initialize=M_value)
        
        # Variables as defined in the PDF formulation
        
        # vᵢ: Squared voltage magnitude at bus i in equations (26), (29), (30)
        model.V_m_sqr = Var(model.buses, model.times, within=NonNegativeReals, initialize=1.0)
        
        # Pᵢⱼ: Active power flow on line (i,j) in equations (27), (29)
        model.P_flow = Var(model.lines, model.times, initialize=0)
        
        # Qᵢⱼ: Reactive power flow on line (i,j) in equations (28), (29)
        model.Q_flow = Var(model.lines, model.times, initialize=0)
        
        # ℓᵢⱼ: Squared current magnitude on line (i,j) in equations (25), (26), (29)
        model.l_squared = Var(model.lines, model.times, within=NonNegativeReals, initialize=1)
        
        # yⱼ: Binary variable indicating if bus j is energized (equation 33)
        model.y = Var(model.partitioned_buses, within=Binary, initialize=1)

        # Slack variable for feasibility (not in the theoretical formulation, but needed for practical implementation)
        model.voltage_slack = Var(model.lines, model.times, within=NonNegativeReals, initialize=0)
        model.voltage_slack_penalty = Param(initialize=1000, mutable=True)
        
        # Line direction maps
        line_starts = {row["name"]: self.bus_dict.get(self.net.line.at[idx, "from_bus"])
                for idx, row in self.net.line.iterrows()
                if ( self.bus_dict.get(self.net.line.at[idx, "from_bus"]) is not None and 
                    self.bus_dict.get(self.net.line.at[idx, "from_bus"]) in self.voltages_bus.index ) }
        line_ends = { row["name"]: self.bus_dict.get(row["to_bus"])
                    for idx, row in self.net.line.iterrows()
                    if self.bus_dict.get(row["to_bus"]) is not None and self.bus_dict.get(row["to_bus"]) in self.voltages_bus.index
                }
        model.line_start = Param(
            model.lines,
            within=Any,
            initialize=lambda m, l: line_starts.get(l, None)
        )
        model.line_end = Param(
            model.lines,
            within=Any,
            initialize=lambda m, l: line_ends.get(l, None)
        )
        
        # xᵢⱼ: Binary variable for line status in equations (6), (32)
        model.line_status = Var(model.lines, domain=Binary, initialize=1)
        
        # Switch definitions
        line_switches = self.switch_df[self.switch_df['et'] == 'l']
        model.switches = Set(initialize=list(line_switches.index), ordered=True)
        model.switch_mask = Param(
            model.switches,
            within=Binary,
            initialize=lambda m, s: 1,
            mutable=True
        )
        model.switch_status = Var(
            model.switches,
            within=Binary,
            initialize=lambda m, s: 1 if self.switch_df.at[s, 'closed'] else 0
        )
        model.switch_initial = Param(
            model.switches,
            initialize=lambda m, s: 1 if self.switch_df.at[s, 'closed'] else 0
        )  

        model.active_slack_penalty = Param(initialize=1000, mutable=True)
        model.reactive_slack_penalty = Param(initialize=1000, mutable=True) 
        
        # Active and Reactive power balance slack
        model.p_slack = Var(model.buses, model.times, domain=Reals, initialize=0)
        model.q_slack = Var(model.buses, model.times, domain=Reals, initialize=0)   
                
        # Objective function  
        # min Σ(i,j)∈Ωl rij * ℓij (equation 25)
        def objective_rule_socp(m):
            # Loss term from resistive losses
            loss_term = sum(m.rl_mOhm[l] * m.l_squared[l, t] * 1e-3 for l in m.lines for t in m.times)

            # Voltage slack term (already in your formulation)
            voltage_slack_term = sum(m.voltage_slack_penalty * m.voltage_slack[l, t] for l in m.lines for t in m.times)

            # Penalties for slack in power balances (using squared slack for smoother penalization)
            reactive_slack_term = sum(m.reactive_slack_penalty * (m.q_slack[b, t] ** 2) for b in m.buses for t in m.times)
            active_slack_term = sum(m.active_slack_penalty * (m.p_slack[b, t] ** 2) for b in m.buses for t in m.times)
            # Optional switch penalty term if toggled on
            switch_penalty_term = 0
            if self.toggles["include_switch_penalty"]:
                switch_penalty_term = self.switch_penalty * sum(
                    (1 - m.switch_initial[s]) * m.switch_status[s] + m.switch_initial[s] * (1 - m.switch_status[s])
                    for s in m.switches
                )
            return loss_term + voltage_slack_term + reactive_slack_term + active_slack_term + switch_penalty_term
        model.objective = Objective(rule=objective_rule_socp, sense=minimize)
        # def objective_rule_socp_with_switch_penalty(m):
        #     loss_term = sum(m.rl_mOhm[l] * m.l_squared[l, t] * 1e-3 for l in m.lines for t in m.times)
        #     voltage_slack_term = sum(m.voltage_slack_penalty * m.voltage_slack[l, t] for l in m.lines for t in m.times)
        #     reactive_slack_term = sum(m.reactive_slack_penalty * (m.q_slack[b, t] ** 2) for b in m.buses for t in m.times)
        #     active_slack_term = sum(m.active_slack_penalty * (m.p_slack[b, t] ** 2) for b in m.buses for t in m.times)
            
        #     # Switching cost: add a small cost for any change in switch status.
        #     switch_penalty_term = self.switch_penalty * sum(
        #         abs(m.switch_status[s] - m.switch_initial[s]) for s in m.switches
        #     )
            
        #     return loss_term + voltage_slack_term + reactive_slack_term + active_slack_term + switch_penalty_term

        # model.objective = Objective(rule=objective_rule_socp_with_switch_penalty, sense=minimize)
 

        # Active power balance: Σj∈Ωbi Pij - Σj∈Ωbi Pji = PSi - PDi ∀i ∈ Ωb (equation 27)
        def active_power_balance_rule(m, b, t):
            if b in self.substations:
                return Constraint.Skip
            else:
                outflow = sum(m.P_flow[l, t] for l in m.lines if m.line_start[l] == b)
                inflow = sum(m.P_flow[l, t] for l in m.lines if m.line_end[l] == b)
                # Allow slack in active power balance
                return outflow - inflow - self.bus_injection[b] == m.p_slack[b, t]

        # Reactive power balance: Σj∈Ωbi Qij - Σj∈Ωbi Qji = QSi - QDi ∀i ∈ Ωb (equation 28)
        def reactive_power_balance_rule(m, b, t): 
            if b in self.substations:
                return Constraint.Skip
            else:
                outflow_q = sum(m.Q_flow[l, t] for l in m.lines if m.line_start[l] == b) 
                inflow_q = sum(m.Q_flow[l, t] for l in m.lines if m.line_end[l] == b) 
                q_injection = self.reactive_bus_injection.get(b, 0) 
                # Allow a slack term on the right-hand side 
                return outflow_q - inflow_q - q_injection == m.q_slack[b, t]

        if toggles["include_power_balance_constraint"]:
            model.ActivePowerBalance = Constraint(model.buses, model.times, rule=active_power_balance_rule)
            model.ReactivePowerBalance = Constraint(model.buses, model.times, rule=reactive_power_balance_rule)
        
        # SOCP Cone Constraint
        # ||(2Pij, 2Qij, vi - ℓij)||2 ≤ vi + ℓij ∀(ij) ∈ Ωl (equation 29)
        def socp_cone_rule(m, l, t):
            from_bus = m.line_start[l]
            if from_bus is None:
                # Skip the constraint if the starting bus is undefined
                return Constraint.Skip
            return (4*m.P_flow[l, t]**2 + 4*m.Q_flow[l, t]**2 +
                    (m.V_m_sqr[from_bus, t] - m.l_squared[l, t])**2) <= (m.V_m_sqr[from_bus, t] + m.l_squared[l, t])**2
        
        model.SOCP_Cone = Constraint(model.lines, model.times, rule=socp_cone_rule)
        
        # Voltage Drop Constraint
        # vj = vi - 2(rijPij + xᵢⱼQij) + (r²ij + x²ij)ℓij ∀(ij) ∈ Ωl (equation 26)
        # def voltage_drop_rule(m, l, t):
        #     i = m.line_start[l]
        #     j = m.line_end[l]
        #     R = m.rl_mOhm[l] * 1e-3  # mOhm to ohm conversion
        #     X = m.xl_mOhm[l] * 1e-3
        #     return (m.V_m_sqr[j, t] + m.voltage_slack[l, t] == 
        #             m.V_m_sqr[i, t] - 2*(R * m.P_flow[l, t] + X * m.Q_flow[l, t]) + (R**2 + X**2) * m.l_squared[l, t])
        def voltage_drop_rule(m, l, t):
            i = m.line_start[l]
            j = m.line_end[l]
            # Skip this constraint if either bus is undefined
            if i is None or j is None:
                return Constraint.Skip
            R = m.rl_mOhm[l] * 1e-3  # mOhm to ohm conversion
            X = m.xl_mOhm[l] * 1e-3
            return (m.V_m_sqr[j, t] + m.voltage_slack[l, t] ==
                    m.V_m_sqr[i, t] - 2*(R * m.P_flow[l, t] + X * m.Q_flow[l, t]) + (R**2 + X**2) * m.l_squared[l, t])
        if toggles["include_voltage_drop_constraint"]:
            model.VoltageDrop = Constraint(model.lines, model.times, rule=voltage_drop_rule)
        
        # Voltage bounds
        # v ≤ vi ≤ v̄ ∀i ∈ Ωb (equation 30)
        # def voltage_upper_bound(m, b, t):
        #     return m.V_m_sqr[b, t] <= m.V_overline[b] ** 2
        
        # def voltage_lower_bound(m, b, t):
        #     return m.V_m_sqr[b, t] >= m.V_underline[b] ** 2
        def voltage_upper_bound(m, b, t):
            if b in m.partitioned_buses:
                # When bus is off (y[b]==0), allow the upper bound to relax.
                return m.V_m_sqr[b, t] <= m.V_overline[b] ** 2 + (1 - m.y[b]) * m.big_M
            else:
                return m.V_m_sqr[b, t] <= m.V_overline[b] ** 2

        def voltage_lower_bound(m, b, t):
            if b in m.partitioned_buses:
                # When bus is off, the lower bound is relaxed.
                return m.V_m_sqr[b, t] >= m.V_underline[b] ** 2 - (1 - m.y[b]) * m.big_M
            else:
                return m.V_m_sqr[b, t] >= m.V_underline[b] ** 2
        if toggles["include_voltage_bounds_constraint"]:
            model.VoltageUpper = Constraint(model.buses, model.times, rule=voltage_upper_bound)
            model.VoltageLower = Constraint(model.buses, model.times, rule=voltage_lower_bound)
        
        # def fix_line_status(m, l):
        #     if l not in self.lines_with_switches:
        #         return m.line_status[l] == 1
        #     else:
        #         return Constraint.Skip
        def fix_line_status(m, l):
            # Option: Only fix ON if the line is NOT switchable AND was initially in service 
            # (Requires knowing initial state, e.g., from self.net.line['in_service'])
            # Example assuming line_df has 'name' and 'in_service' from original data:
            line_data = self.line_df[self.line_df['name'] == l].iloc[0] 
            is_initially_in_service = line_data.get('in_service', True) # Default to True if column missing

            if l not in self.lines_with_switches:
                if is_initially_in_service:
                    return m.line_status[l] == 1 # Fix ON only if initially ON and no switch
                else:
                    #return m.line_status[l] == 0 # Fix OFF if initially OFF and no switch
                    return Constraint.Skip # Or let it be decided by optimization if initially OFF
            else: # Line has a switch, status is variable
                return Constraint.Skip

        model.FixLineStatus = Constraint(model.lines, rule=fix_line_status)
        
        def slack_link_rule(m, l, t):
            # Only apply to buses in the partitioned set where y is defined.
            if m.line_end[l] in m.partitioned_buses:
                # If bus is energized (y==1) then slack must be near zero.
                return m.voltage_slack[l, t] <= m.big_M * (1 - m.y[m.line_end[l]])
            else:
                return Constraint.Skip

        model.SlackLink = Constraint(model.lines, model.times, rule=slack_link_rule)
        
        
        # Radiality Constraints
        if toggles["include_radiality_constraints"]:
            model.substations = Set(initialize=list(self.substations))
            
            # Additional variable for flow-based radiality verification
            model.flow = Var(model.lines, domain=NonNegativeReals, initialize=0)
        
            # Line status constraints based on switch status - maps to equation (32)
            def line_status_rule(m, l):
                if l in self.lines_with_switches:
                    return m.line_status[l] == sum(m.switch_status[s] for s in self.lines_with_switches[l])
                else:
                    return Constraint.Skip
            
            model.LineStatusConstraint = Constraint(model.lines, rule=line_status_rule)
            
            # Bus-line connection constraints
            # xij ≤ yj, xji ≤ yj ∀(ij) ∈ Ωl, ∀j ∈ Ωbp (equation 34)
            def line_bus_connection_rule1(m, l, b):
                if m.line_start[l] == b and b in m.partitioned_buses:
                    return m.line_status[l] <= m.y[b]
                else:
                    return Constraint.Skip
            
            model.LineBusConnection1 = Constraint(model.lines, model.partitioned_buses, rule=line_bus_connection_rule1)
            
            def line_bus_connection_rule2(m, l, b):
                if m.line_end[l] == b and b in m.partitioned_buses:
                    return m.line_status[l] <= m.y[b]
                else:
                    return Constraint.Skip
            
            model.LineBusConnection2 = Constraint(model.lines, model.partitioned_buses, rule=line_bus_connection_rule2)
            
            # Bus connectivity constraint
            # Σ(ij)∈Ωl xij + Σ(ji)∈Ωl xji ≥ 2yj ∀k ∈ Ωbp (equation 35)
            # def bus_connectivity_rule(m, b):
            #     if b in m.partitioned_buses:
            #         # Collect all incident lines (both where b is start or end)
            #         incident_lines = [l for l in m.lines if m.line_start[l] == b or m.line_end[l] == b]
            #         degree = len(incident_lines)
            #         # For degree 1, require the single incident line be active;
            #         # For degree >=2, require at least 2 active connections.
            #         required = m.y[b] if degree == 1 else 2 * m.y[b]
            #         return sum(m.line_status[l] for l in incident_lines) >= required
            #     else:
            #         return Constraint.Skip
            def bus_connectivity_rule(m, b): 
                if b in m.partitioned_buses: 
                    # Gather all incident lines (both originating from and ending at bus b) 
                    incident_lines = [l for l in m.lines if m.line_start[l] == b or m.line_end[l] == b] 
                    # Relax the requirement: if the bus is energized (y[b]==1), require at least one connection. 
                    return sum(m.line_status[l] for l in incident_lines) >= m.y[b] 
                else: 
                    return Constraint.Skip
            
            model.BusConnectivityConstraint = Constraint(model.partitioned_buses, rule=bus_connectivity_rule)
            
            # Spanning tree or radiality constraint
            # Σ(ij)∈Ωl xij = nb - 1 - Σj∈Ωbp(1 - yj) (equation 36)
            if toggles["use_spanning_tree_radiality"]:
                def spanning_tree_rule(m):
                    inactive_buses = sum(1 - m.y[b] for b in m.partitioned_buses)
                    return sum(m.line_status[l] for l in m.lines) == len(m.buses) - 1 - inactive_buses
                
                model.SpanningTree = Constraint(rule=spanning_tree_rule)
            else:
                # Alternative flow-based radiality formulation
                # Big-M flow activation
                def big_m_flow_rule(m, l, t):
                    return m.flow[l] <= m.big_M * m.line_status[l]
                
                model.BigMFlow = Constraint(model.lines, model.times, rule=big_m_flow_rule)
            
                # Flow balance constraints for radiality verification
                def flow_balance_rule(m, b):
                    inflow = sum(m.flow[l] for l in m.lines if m.line_end[l] == b)
                    outflow = sum(m.flow[l] for l in m.lines if m.line_start[l] == b)
                    if b in m.substations:
                        total_demand = sum(m.radial_demand[i] for i in m.buses if i not in m.substations)
                        return outflow - inflow == total_demand
                    else:
                        return inflow - outflow >= m.radial_demand[b]
                
                model.RadialityFlowBalance = Constraint(model.buses, rule=flow_balance_rule)
            
                def flow_activation_rule(m, l):
                    if l not in self.lines_with_switches:
                        return Constraint.Skip
                    switch_sum = sum(m.switch_status[s] for s in self.lines_with_switches.get(l, []))
                    return m.flow[l] <= (len(m.buses) - 1) * switch_sum
                
                model.RadialityFlowActivation = Constraint(model.lines, rule=flow_activation_rule)
                    
        # Power Loss Expression - rij * ℓij from equation (25)
        model.P_loss = Expression(model.lines, model.times, rule=lambda m, l, t: m.rl_mOhm[l] * m.l_squared[l, t] * 1e-3)

        return model
    def solve(self, solver='gurobi', verbose=True, model=None, **solver_options):
        """
        Solves the optimization model for distribution network reconfiguration.
        
        Args:
            solver (str): Name of the solver to use (default 'gurobi').
            verbose (bool): If True, prints detailed solver output.
            model: Optional pre-created Pyomo model. If provided, uses this model.
                If None and self.model is None, a new model is created.
            **solver_options: Additional options passed to the solver interface.
        
        Returns:
            Optimization results dictionary, or None if solving fails.
        """
        # Use provided model or create one if necessary
        if model is not None:
            self.model = model
        if self.model is None:
            self.logger.info(f"Creating optimization model for {getattr(self, 'net_name', 'default')}...")
            try:
                self.model = self.create_model()
            except Exception as create_e:
                self.logger.error(f"Failed to create model: {create_e}", exc_info=True)
                return None

        self.logger.info(f"Starting optimization for {getattr(self, 'net_name', 'default')}...")
        start_time = time.time()

        try:
            if solver.lower() == 'gurobi':
                try:
                    import gurobipy as gp
                    self.logger.info("Using Gurobi solver directly via gurobipy...")
                    lp_filename = f"model_{getattr(self, 'graph_id', 'default')}.lp"
                    self.model.write(lp_filename, io_options={"symbolic_solver_labels": True})
                    self.logger.info(f"Wrote model to {lp_filename}")
                    gurobi_model = gp.read(lp_filename)
                    defaults = {
                        "MIPGap": 0.001,
                        "TimeLimit": 3000,
                        "MIPFocus": 3,
                        "Threads": 8,
                        "OptimalityTol": 1e-3,
                        "FeasibilityTol": 1e-3,
                        "NonConvex": 2,
                        "NumericFocus": 3,
                        "BarHomogeneous": 1,

                    }
                    for key, default_value in defaults.items():
                        param_value = solver_options.get(key, default_value)
                        gurobi_model.setParam(key, param_value)
                        self.logger.info(f"Setting Gurobi parameter: {key} = {param_value}")
                    gurobi_model.optimize()

                    if gurobi_model.Status == gp.GRB.OPTIMAL:
                        self.logger.info(f"Optimal solution found with objective: {gurobi_model.ObjVal}")
                    elif gurobi_model.Status in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD]:
                        self.logger.warning("Model is infeasible or unbounded. Attempting IIS computation...")
                        gurobi_model.computeIIS()
                        iis_filename = f"SOCP_ISS_{self.graph_id}.ilp"
                        gurobi_model.write(iis_filename)
                        self.logger.info(f"Wrote IIS information to {iis_filename}")
                        try:
                            with open(iis_filename, "r") as iis_file:
                                iis_details = iis_file.read()
                            self.logger.info(f"IIS details:\n{iis_details}")
                        except Exception as e:
                            self.logger.error(f"Failed to read IIS file: {e}")
                    else:
                        self.logger.info(f"Gurobi solver status: {gurobi_model.Status}")
                    # You might choose to extract results from gurobipy here.
                except ImportError:
                    self.logger.warning("Gurobipy not available. Falling back to SolverFactory...")
                    opt = SolverFactory(solver)
            else:
                opt = SolverFactory(solver)

            if solver.lower() != 'gurobi':
                # Use SolverFactory branch. Set default options and update with any solver_options.
                current_options = solver_options.copy()
                if solver.lower() == 'gurobi':
                    defaults = {
                        "mipgap": 0.001,
                        "TimeLimit": 300,
                        "MIPFocus": 3,
                        "Nonconvex": 2,
                        "Threads": 8,
                        "OptimalityTol": 1e-6,
                        "FeasibilityTol": 1e-6,
                        "IntFeasTol": 1e-6
                    }
                    for key, value in defaults.items():
                        current_options.setdefault(key, value)
                if current_options:
                    self.logger.info("Applying solver options:")
                    for key, value in current_options.items():
                        opt.options[key] = value
                        self.logger.info(f"  Setting solver option: {key} = {value}")

                self.solver_results = opt.solve(self.model, tee=verbose)
                self.logger.info("Solver finished.")

                # --- Check Solver Results and Debug Infeasibility ---
                if (self.solver_results is not None and 
                    hasattr(self.solver_results, 'solver') and 
                    self.solver_results.solver is not None):
                    term_cond = self.solver_results.solver.termination_condition
                    if term_cond in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]:
                        self.logger.warning("Model reported as infeasible or unbounded. Attempting IIS computation...")
                        lp_filename = f"data_generation/logs/infeasible_model_{getattr(self, 'graph_id', 'default')}.lp"
                        try:
                            self.model.write(lp_filename, io_options={'symbolic_solver_labels': True})
                            self.logger.info(f"Wrote potentially infeasible model to {lp_filename}")
                        except Exception as write_e:
                            self.logger.error(f"Failed to write model file: {write_e}")
                        if solver.lower() == 'gurobi':
                            # In the SolverFactory branch, we may not have direct access to computeIIS.
                            self.logger.warning("To compute IIS with Gurobi, run:")
                            self.logger.warning(f"  gurobi_cl ResultFile={lp_filename}.ilp {lp_filename}")
                            self.logger.warning("(Requires Gurobi command line tools installed and in PATH)")
                        else:
                            self.logger.warning(f"IIS computation not automatically triggered for solver '{solver}'. Check solver documentation.")

                    self.logger.info(f"  Status: {self.solver_results.solver.status}")
                    self.logger.info(f"  Termination Condition: {term_cond}")

                    if (self.solver_results.solver.status == pyo.SolverStatus.ok and 
                        term_cond in [TerminationCondition.optimal, TerminationCondition.feasible, TerminationCondition.maxTimeLimit]):
                        obj_val = pyo_value(self.model.objective)
                        self.logger.info(f"Objective value (p.u. loss sum): {obj_val:.6f}")
                        if term_cond == TerminationCondition.maxTimeLimit:
                            self.logger.warning("Solver reached time limit, solution may be suboptimal.")
                        elif term_cond != TerminationCondition.optimal:
                            self.logger.warning(f"Solver found a feasible but non-optimal solution ({term_cond}).")
                    elif term_cond == TerminationCondition.infeasible:
                        self.logger.warning("Problem declared infeasible by the solver.")
                    else:
                        self.logger.error(f"Solver failed or reported non-optimal status: {term_cond}")
                else:
                    self.logger.error("No valid solver results were obtained.")

        except Exception as e:
            self.logger.error(f"An error occurred during solving with {solver}: {e}", exc_info=True)
            self.solver_results = None

        self.optimization_time = time.time() - start_time
        self.logger.info(f"Optimization completed in {self.optimization_time:.2f} seconds")
        if getattr(self, 'include_switches', False):
            self.num_switches_changed = self._count_changed_switches()
            self.logger.info(f"Number of switches changed: {self.num_switches_changed}")
        

        self.model.dual = Suffix(direction=Suffix.IMPORT)

        # Define a small tolerance
        tolerance = 1e-6

        # Log nonzero slack values for active and reactive power balance
        for b in self.model.buses:
            for t in self.model.times:
                p_slack_val = pyo_value(self.model.p_slack[b, t])
                q_slack_val = pyo_value(self.model.q_slack[b, t])
                if abs(p_slack_val) > tolerance or abs(q_slack_val) > tolerance:
                    self.logger.debug(f"Bus {b}, time {t}: p_slack = {p_slack_val:.4f}, q_slack = {q_slack_val:.4f}")

        # Log dual variables for power balance constraints
        # Active power balance (only log for buses not in substations)
        for b in self.model.buses:
            for t in self.model.times:
                if b not in self.substations:
                    active_con = self.model.ActivePowerBalance[b, t]
                    active_dual = self.model.dual.get(active_con, None)
                    self.logger.debug(f"Bus {b}, time {t}: Dual of ActivePowerBalance = {active_dual}")

        # Reactive power balance duals
        for b in self.model.buses:
            for t in self.model.times:
                reactive_con = self.model.ReactivePowerBalance[b, t]
                reactive_dual = self.model.dual.get(reactive_con, None)
                self.logger.debug(f"Bus {b}, time {t}: Dual of ReactivePowerBalance = {reactive_dual}")
        self.extract_results()

        
        return self.optimized_results

    
    def _count_changed_switches(self):
        changes = 0
        if not self.include_switches or self.model is None:
            return changes
        for s in self.model.switches:
            try:
                initial = bool(self.initial_switch_status.at[s])
                optimized = bool(round(pyo_value(self.model.switch_status[s])))
                if initial != optimized:
                    changes += 1
            except:
                pass
        return changes

    def extract_results(self):
        if not hasattr(self.model, "P_loss"):
            raise RuntimeError("Model is infeasible or was not solved correctly; 'P_loss' does not exist.")

        if self.model is None:
            return
        results = {
            "optimization_time": self.optimization_time,
            "num_switches": len(self.model.switches) if self.include_switches else 0,
            "num_switches_changed": self.num_switches_changed if self.include_switches else 0,
            "objective_value": pyo_value(self.model.objective),
            "power_loss": sum(pyo_value(self.model.P_loss[l, 0]) for l in self.model.lines),
        }
        if self.include_switches:
            switch_results = {}
            for s in self.model.switches:
                try:
                    initial = bool(self.initial_switch_status.at[s])
                    optimized = bool(round(pyo_value(self.model.switch_status[s])))
                    switch_results[s] = {
                        "initial": initial,
                        "optimized": optimized,
                        "changed": initial != optimized
                    }
                except:
                    pass
            results["switches"] = switch_results
        voltage_profiles = {}
        for b in self.model.buses:
            try:
                voltage_profiles[b] = np.sqrt(pyo_value(self.model.V_m_sqr[b, 0]))
            except:
                pass
        results["voltage_profiles"] = voltage_profiles
        self.optimized_results = results
        return results

    def update_network(self):
        if not self.include_switches or self.model is None:
            print("Switch optimization is disabled or model not solved. Network not updated.")
            return self.net
        optimized_statuses = {}
        for s in self.model.switches:
            try:
                switch_status = pyo_value(self.model.switch_status[s])
                optimized_statuses[s] = bool(round(switch_status))
            except:
                print(f"Could not get value for switch {s}")
        for s, status in optimized_statuses.items():
            self.net.switch.at[s, 'closed'] = status
        print("Network configuration updated with optimized switch statuses.")
        return self.net

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
        if self.include_switches:
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
        if not self.include_switches or self.optimized_results is None:
            print("Switch optimization results not available for plotting.")
            return
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