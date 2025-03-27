import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pint import UnitRegistry
ureg = UnitRegistry()

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
    value
)
from pyomo.util.infeasible import log_infeasible_constraints


class SOCP_class:
    """
    This class implements Second Order Conic Programming optimization for 
    distribution network reconfiguration.
    """

    def __init__(
        self,
        net,
        net_name: str = "Network",
        switch_penalty: float = 0.01,
        ):
        self.net = net
        self.net_name = net_name
        if hasattr(self.net, 'res_line') and hasattr(self.net.res_line, 'pl_mw'):
            active_power_loss = self.net.res_line.pl_mw
            self.total_active_power_loss = sum(active_power_loss)
        else:
            self.total_active_power_loss = None
        self.bus_injection = None

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
        line_df["r_ohm"] = line_df.apply(
            lambda row: (row["r_ohm_per_km"] * (ureg.ohm/ureg.km) * row["length_km"] * ureg.km).to(ureg.ohm).magnitude,
            axis=1
        )
        # Compute reactance in ohm similarly:
        line_df["x_ohm"] = line_df.apply(
            lambda row: (row["x_ohm_per_km"] * (ureg.ohm/ureg.km) * row["length_km"] * ureg.km).to(ureg.ohm).magnitude,
            axis=1
        )
        frequency = 50  # Hz
        line_df["B_Siemens_shunt"] = line_df.apply(
            lambda row: row["c_nf_per_km"] * row["length_km"] * 2 * np.pi * frequency / 10**9, axis=1
        )
        def susceptance_cal(R, X):
            return -1 / X if R == 0 else -X / (R**2 + X**2)
        line_df["B_transmission_s"] = line_df.apply(lambda row: susceptance_cal(row["r_ohm"], row["x_ohm"]), axis=1)

        # Bus susceptance 
        bus_susceptance = {bus: 0 for bus in self.net.bus.index}
        for idx, row in self.net.line.iterrows():
            B = row["c_nf_per_km"] * row["length_km"] * 2 * np.pi * frequency * 1e-9
            bus_susceptance[row["from_bus"]] += B
            bus_susceptance[row["to_bus"]] += B
        B_df = pd.DataFrame({
            "bus_n": list(bus_susceptance.keys()),
            "bus_B_s": list(bus_susceptance.values()),
            "bus_name": [bus_dict[i] for i in bus_susceptance.keys()]
        })

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

        # Save processed data as attributes
        self.line_df = line_df
        self.bus_df = bus_df
        self.bus_dict = bus_dict
        self.voltages_bus = voltages_bus
        self.B_df = B_df


    def create_model(self, toggles=None):
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
        model.buses = Set(initialize=list(self.voltages_bus.index), ordered=True)
        model.lines = Set(initialize=list(self.line_df["name"]), ordered=True)
        model.times = Set(initialize=[0], ordered=True)
        
        #Voltage parameters 
        model.V_overline = Param(
            model.buses,
            initialize=lambda m, b: ((self.voltages_bus.at[b, 'vn_kv'] * ureg.kV * 1000 * 1.05)
                                        .to(ureg.volt).magnitude)
        )
        model.V_underline = Param(
            model.buses,
            initialize=lambda m, b: ((self.voltages_bus.at[b, 'vn_kv'] * ureg.kV * 1000 * 0.95)
                                        .to(ureg.volt).magnitude)
        )
        model.xl_mOhm = Param(
            model.lines,
            within=NonNegativeReals,
            initialize=lambda m, l: ((self.line_df.set_index("name").at[l, "x_ohm"] * ureg.ohm)
                                        .to(ureg.mohm).magnitude)
        )
        model.rl_mOhm = Param(
            model.lines,
            within=NonNegativeReals,
            initialize=lambda m, l: ((self.line_df.set_index("name").at[l, "r_ohm"] * ureg.ohm)
                                        .to(ureg.mohm).magnitude)
        )
        
        model.radial_demand = Param(model.buses, initialize=lambda m, b: 0 if b in self.substations else 1)
        
        expected_max_flow = sum(abs(val) for val in self.bus_injection.values())
        print("Expected maximum flow (used for Big‑M):", expected_max_flow)
        M_value = expected_max_flow * 1.1  # Safety factor
        model.big_M = Param(initialize=M_value)
        
        # Variables 
        model.V_m_sqr = Var(model.buses, model.times, within=NonNegativeReals, initialize=1.0)
        model.P_flow = Var(model.lines, model.times, initialize=0)
        model.Q_flow = Var(model.lines, model.times, initialize=0)
        model.l_squared = Var(model.lines, model.times, within=NonNegativeReals, initialize=1)

        model.voltage_slack = Var(model.lines, model.times, within=NonNegativeReals, initialize=0)
        model.voltage_slack_penalty = Param(initialize=1000, mutable=True)
        
        # Line direction maps 
        line_starts = {row["name"]: self.bus_dict.get(self.net.line.at[idx, "from_bus"])
               for idx, row in self.net.line.iterrows() if self.bus_dict.get(self.net.line.at[idx, "from_bus"]) is not None}
        line_ends = {row["name"]: self.bus_dict[row["to_bus"]] for _, row in self.net.line.iterrows()}
        model.line_start = Param(model.lines, initialize=line_starts)
        model.line_end = Param(model.lines, initialize=line_ends)
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
        
        # Objective function 
        # def objective_rule_socp(m):
        #     loss_term = sum(m.rl_mOhm[l] * m.l_squared[l, t] for l in m.lines for t in m.times)
        #     penalty_term = self.switch_penalty * sum(
        #             (1 - m.switch_initial[s]) * m.switch_status[s] + m.switch_initial[s] * (1 - m.switch_status[s])
        #             for s in m.switches
        #     )
        #     if toggles["include_switch_penalty"]:
        #         return loss_term + penalty_term
        #     else:
        #         return loss_term

        # add slack
        def objective_rule_socp(m):
            loss_term = sum(m.rl_mOhm[l] * m.l_squared[l, t] for l in m.lines for t in m.times)
            slack_term = sum(m.voltage_slack_penalty * m.voltage_slack[l, t] for l in m.lines for t in m.times)
            penalty_term = self.switch_penalty * sum(
                (1 - m.switch_initial[s]) * m.switch_status[s] + m.switch_initial[s] * (1 - m.switch_status[s])
                for s in m.switches
            )
            if toggles["include_switch_penalty"]:
                return loss_term + slack_term + penalty_term
            else:
                return loss_term + slack_term
        model.objective = Objective(rule=objective_rule_socp, sense=minimize)
        
        # Power Balance Constraint 
        def power_balance_rule(m, b, t):
            outflow = sum(m.P_flow[l, t] for l in m.lines if m.line_start[l] == b)
            inflow  = sum(m.P_flow[l, t] for l in m.lines if m.line_end[l] == b)
            expr = outflow - inflow - self.bus_injection[b]
            # If the expression is a number and is zero, skip the constraint
            if isinstance(expr, (int, float)):
                if expr == 0:
                    return Constraint.Skip
                else:
                    return expr == 0
            else:
                # Otherwise, return the symbolic equality
                return expr == 0
        if toggles["include_power_balance_constraint"]:
            model.PowerBalance = Constraint(model.buses, model.times, rule=power_balance_rule)
        
        # SOCP Cone Constraint 
        def socp_cone_rule(m, l, t):
            from_bus = m.line_start[l]
            return (m.P_flow[l, t]**2 + m.Q_flow[l, t]**2) <= m.V_m_sqr[from_bus, t] * m.l_squared[l, t]
        model.SOCP_Cone = Constraint(model.lines, model.times, rule=socp_cone_rule)
        
        # Voltage Drop Constraint 
        # def voltage_drop_rule(m, l, t):
        #     i = m.line_start[l]
        #     j = m.line_end[l]
        #     R = m.rl_mOhm[l] * 1e-3  # Convert mOhm to ohm
        #     X = m.xl_mOhm[l] * 1e-3  # Convert mOhm to ohm
        #     return m.V_m_sqr[j, t] == m.V_m_sqr[i, t] - 2 * (R * m.P_flow[l, t] + X * m.Q_flow[l, t]) + (R**2 + X**2) * m.l_squared[l, t]

        #add slack
        def voltage_drop_rule(m, l, t):
            i = m.line_start[l]
            j = m.line_end[l]
            R = m.rl_mOhm[l] * 1e-3  # mOhm to ohm conversion
            X = m.xl_mOhm[l] * 1e-3
            return m.V_m_sqr[j, t] + m.voltage_slack[l, t] == m.V_m_sqr[i, t] - 2*(R * m.P_flow[l, t] + X * m.Q_flow[l, t]) \
                   + (R**2 + X**2)* m.l_squared[l, t]

        if toggles["include_voltage_drop_constraint"]:
            model.VoltageDrop = Constraint(model.lines, model.times, rule=voltage_drop_rule)
        
        # Voltage bounds 
        def voltage_upper_bound(m, b, t):
            return m.V_m_sqr[b, t] <= m.V_overline[b] ** 2
        def voltage_lower_bound(m, b, t):
            return m.V_m_sqr[b, t] >= m.V_underline[b] ** 2
        if toggles["include_voltage_bounds_constraint"]:
            model.VoltageUpper = Constraint(model.buses, model.times, rule=voltage_upper_bound)
            model.VoltageLower = Constraint(model.buses, model.times, rule=voltage_lower_bound)
        
        # Radiality Constraints
        if toggles["include_radiality_constraints"]:
            model.substations = Set(initialize=list(self.substations))
            model.flow = Var(model.lines, domain=NonNegativeReals, initialize=0)
        
            def line_status_rule(m, l):
                if l in self.lines_with_switches:
                    return m.line_status[l] == sum(m.switch_status[s] for s in self.lines_with_switches[l])
                else:
                    return Constraint.Skip
            model.LineStatusConstraint = Constraint(model.lines, rule=line_status_rule)
            
            # spanning tree fomrulation 
            if toggles["use_spanning_tree_radiality"]:
                def spanning_tree_rule(m):
                    return sum(m.line_status[l] for l in m.lines) == len(m.buses) - 1
                model.SpanningTree = Constraint(rule=spanning_tree_rule)
            else:
                # Big-M flow activationr
                def big_m_flow_rule(m, l, t):
                    return m.flow[l] <= m.big_M * m.line_status[l]
                model.BigMFlow = Constraint(model.lines, model.times, rule=big_m_flow_rule)
            
                # flow balance: non-substation nodes must receive at least one unit
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
                
        # Power Loss Expression 
        model.P_loss = Expression(model.lines, model.times, rule=lambda m, l, t: m.rl_mOhm[l] * m.l_squared[l, t])

        return model


    def solve(self, model=None):
        if model is not None:
            self.model = model

        print(f"Starting optimization for {self.net_name}...")
        start_time = time.time()

        try:
            import gurobipy as gp
            print("Using Gurobi solver directly...")
            self.model.write("model.lp", io_options={"symbolic_solver_labels": True})
            gurobi_model = gp.read("model.lp")
            gurobi_model.setParam("MIPGap", 0.001)
            gurobi_model.setParam("TimeLimit", 300)
            gurobi_model.setParam("MIPFocus", 3)
            gurobi_model.setParam("Threads", 8)
            gurobi_model.setParam("OptimalityTol", 1e-5)
            gurobi_model.setParam("FeasibilityTol", 1e-5)
            gurobi_model.setParam("NonConvex", 2)
            gurobi_model.setParam("NumericFocus", 3)
            gurobi_model.optimize()
            if gurobi_model.Status == gp.GRB.OPTIMAL:
                print(f"Optimal solution found with objective: {gurobi_model.ObjVal}")
            elif gurobi_model.Status in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD]:
                print("Model is infeasible or unbounded. Computing IIS...")
                gurobi_model.computeIIS()
                gurobi_model.write("model.ilp")
                with open("model.ilp", "r") as iis_file:
                    iis = iis_file.read()
                print("IIS:\n", iis)
            else:
                print(f"Solver status: {gurobi_model.Status}")
        except ImportError:
            print("Gurobi not available. Using SolverFactory...")
            solver_name = "gurobi"
            try:
                opt = SolverFactory(solver_name)
                opt.options["mipgap"] = 0.001
                opt.options["TimeLimit"] = 300
                opt.options["MIPFocus"] = 3
                opt.options["Nonconvex"] = 2
                opt.options["Threads"] = 8
                opt.options["OptimalityTol"] = 1e-6
                opt.options["FeasibilityTol"] = 1e-6
                opt.options["IntFeasTol"] = 1e-6
                results = opt.solve(self.model, tee=True)
                if results.solver.termination_condition == "infeasible":
                    print("The model is infeasible. Logging infeasible constraints...")
                    log_infeasible_constraints(self.model, log_expression=True, log_variables=True)
                else:
                    print(f"Objective value: {value(self.model.objective)}")
            except Exception as e:
                print(f"Error using {solver_name}: {e}. Trying with ipopt...")
                opt = SolverFactory("ipopt")
                results = opt.solve(self.model, tee=True)
                print(f"Ipopt solver status: {results.solver.termination_condition}")

        self.optimization_time = time.time() - start_time
        print(f"Optimization completed in {self.optimization_time:.2f} seconds")
        if self.include_switches:
            self.num_switches_changed = self._count_changed_switches()
            print(f"Number of switches changed: {self.num_switches_changed}")
        self._extract_results()
        return self.optimized_results

    def _count_changed_switches(self):
        changes = 0
        if not self.include_switches or self.model is None:
            return changes
        for s in self.model.switches:
            try:
                initial = bool(self.initial_switch_status.at[s])
                optimized = bool(round(value(self.model.switch_status[s])))
                if initial != optimized:
                    changes += 1
            except:
                pass
        return changes

    def _extract_results(self):
        if not hasattr(self.model, "P_loss"):
            raise RuntimeError("Model is infeasible or was not solved correctly; 'P_loss' does not exist.")

        if self.model is None:
            return
        results = {
            "optimization_time": self.optimization_time,
            "num_switches": len(self.model.switches) if self.include_switches else 0,
            "num_switches_changed": self.num_switches_changed if self.include_switches else 0,
            "objective_value": value(self.model.objective),
            "power_loss": sum(value(self.model.P_loss[l, 0]) for l in self.model.lines),
        }
        if self.include_switches:
            switch_results = {}
            for s in self.model.switches:
                try:
                    initial = bool(self.initial_switch_status.at[s])
                    optimized = bool(round(value(self.model.switch_status[s])))
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
                voltage_profiles[b] = np.sqrt(value(self.model.V_m_sqr[b, 0]))
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
                switch_status = value(self.model.switch_status[s])
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
