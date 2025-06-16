from typing import Dict, Any, Optional, List
import time
from defusedxml import DTDForbidden
import numpy as np
import pandas as pd
import logging
import networkx as nx
import matplotlib.pyplot as plt
import math
import pandapower as pp
from pathlib import Path
import os 
import sys

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, NonNegativeReals, Reals,
    Binary, Objective, Expression, minimize, Suffix, value as pyo_val,
)

from pyomo.opt import SolverFactory, check_optimal_termination
from pyomo.util.model_size import build_model_size_report
from pint import UnitRegistry


# Add necessary source paths
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from optimization_logging import get_logger


ureg = UnitRegistry()

 
class SOCP_class:
    def __init__(self, net, graph_id: str = "", *,
                 logger: Optional[logging.Logger] = None,
                 switch_penalty: float = 0.0001,                     #0.001
                 slack_penalty: float = 0.1,
                 voltage_slack_penalty = 0.1,
                 load_shed_penalty: float = 100,
                 toggles: Optional[Dict[str, bool]] = None,
                 debug_level= 0,
                 fixed_switches =None,
                 active_bus_mask: Optional[pd.Series] = None) -> None:
        self.voltage_bigM_factor =1
        self.bigM_factor = 1
        self.fixed_switches = fixed_switches
        

        self.net = net
        self.id = graph_id or "unnamed"
        self.logger = logger or logging.getLogger("SOCP_optimizer")
        self.debug_level = debug_level
        self.toggles = toggles or {
            "include_voltage_drop_constraint": True,
            "include_voltage_bounds_constraint": True,
            "include_power_balance_constraint": True,
            "include_radiality_constraints": True,
            "use_spanning_tree_radiality": False,   
            "use_root_flow": False,
            "include_switch_penalty": True,
            "include_cone_constraint": True,
            "all_lines_are_switches": True,
            "allow_load_shed": False,}


        self.logger.info(f"SOCP toggles: {self.toggles}")
        self.switch_penalty = switch_penalty
        self.slack_penalty = slack_penalty
        self.voltage_slack_penalty = voltage_slack_penalty
        self.load_shed_penalty = load_shed_penalty
        self.num_switches_changed = None

        # bus activity mask ------------------------------------------------
        if active_bus_mask is None:
            self.active_bus = pd.Series(True, index=net.bus.index)
        else:
            self.active_bus = pd.Series(active_bus_mask,
                                        index=net.bus.index).fillna(False).astype(bool)

        self.logger.info(f"SOCP active buses: {self.active_bus.sum()} of {len(self.active_bus)}")

        # substations -------------------------------------------------------
        self.substations = set(net.ext_grid.bus.tolist())
        if hasattr(net, "gen") and not net.gen.empty:
            self.substations |= set(net.gen[net.gen.slack].bus.tolist())
        if hasattr(net, "trafo") and not net.trafo.empty:
            self.substations |= set(net.trafo.lv_bus.tolist())
        if not self.substations:
            raise ValueError("No reference / substation buses found in network")

        # switches ----------------------------------------------------------
        self.switch_df = net.switch.copy()
        self.initial_switch_status: Optional[pd.Series] = None 
        self.initial_line_status: Optional[pd.Series] = None

        # containers to be filled in `initialise()`
        self.bus_df: Optional[pd.DataFrame] = None
        self.line_df: Optional[pd.DataFrame] = None
        self.lines_with_sw: Dict[int, List[int]] = {}
        self.lines_wo_sw: List[int] = []
        self.bus_p_inj: Dict[int, float] = {}
        self.bus_q_inj: Dict[int, float] = {}
        self.bigM: Dict[int, float] = {}
        self.v_base_V: Dict[int, float] = {}
        self.S_base_VA: Optional[float] = None

        # Pyomo model & results
        self.model: Optional[ConcreteModel] = None
        self.solver_results = None  
        self.solve_time: Optional[float] = None

        # debugging data containers
        self.constraint_violations: Dict[str, Any] = {}
        self.variable_values: Dict[str, Any] = {}
        self.debug_logs = {}

        # for progressive model building
        self.enabled_constraints = { }
        self.current_validation_step =None


    def initialize(self) -> None:
        S_base_MVA = 1.0         
        self.S_base_VA = S_base_MVA * 1e6     # 1 MVA
        self.bus_df = self.net.bus.copy()

        # ------------------------------------------------ bus table
        self.bus_df["v_base_V_ll"] = (self.bus_df.vn_kv * 1e3)
        
        # Line-to-neutral voltage base (for impedance calculations)
        self.bus_df["v_base_V_ln"] = (self.bus_df.vn_kv * 1e3) / math.sqrt(3)
        self.v_base_V = self.bus_df.v_base_V_ln.to_dict()
        
        # Store both for reference
        self.v_base_V_ll = self.bus_df.v_base_V_ll.to_dict()
        self.v_base_V_ln = self.bus_df.v_base_V_ln.to_dict()
        # ------------------------------------------------ line table
        line = self.net.line.copy()
        # Ω values
        line["r_ohm"] = line.r_ohm_per_km * line.length_km
        line["x_ohm"] = line.x_ohm_per_km * line.length_km

        # per-unit impedances 
        def _rz_pu(row):
            # Z_base = V_base² / S_base   
            Z_base = (self.v_base_V_ln[row.from_bus] ** 2) / self.S_base_VA
            return row.r_ohm / Z_base

        def _xz_pu(row) -> float:
            Z_base = (self.v_base_V_ln[row.from_bus] ** 2) / self.S_base_VA
            return row.x_ohm / Z_base

        line["r_pu"] = line.apply(_rz_pu, axis=1)
        line["x_pu"] = line.apply(_xz_pu, axis=1)   

        self.line_df = line

        # ------------------------------------------------ switches
        sw = self.switch_df 
        self.line_switches = sw[sw.et == "l"]
        self.lines_with_sw = ( self.line_switches.groupby("element").apply(lambda d: list(d.index)).to_dict())
        self.lines_wo_sw = [l for l in line.index if l not in self.lines_with_sw]

        # Calculate initial effective status for all lines
        initial_line_status_dict = {}
        for l in line.index:
            if l in self.lines_with_sw:
                switch_indices = self.lines_with_sw[l]
                initial_line_status_dict[l] = any(sw.loc[s, 'closed'] for s in switch_indices)
            else:
                initial_line_status_dict[l] = 1  
        self.initial_line_status = pd.Series(initial_line_status_dict).astype(int)

        # Store initial switch status only
        if not self.toggles.get('all_lines_are_switches', False):
            self.initial_switch_status = sw['closed'].copy()

        # ------------------------------------------------ bus injections
        self.bus_p_inj = {}
        self.bus_q_inj = {}
        for b in self.bus_df.index:
            p_mw = (self.net.gen[self.net.gen.bus == b].p_mw.sum() - self.net.load[self.net.load.bus == b].p_mw.sum()    )
            q_mvar = ( get_reactive_injection(self.net.gen, b)- get_reactive_injection(self.net.load, b))
            self.bus_p_inj[b] = p_mw / S_base_MVA          
            self.bus_q_inj[b] = q_mvar / S_base_MVA        

        # Debug: log injection values
        if self.debug_level >= 1:
            self.logger.debug(f"Bus {b} injections - P: {self.bus_p_inj[b]:.4f} p.u., Q: {self.bus_q_inj[b]:.4f} p.u.")

        # ------------------------------------------------ Big-M 
        self.bigM = {}
        for l_idx, row in self.line_df.iterrows(): 
            if 'max_i_ka' in row and pd.notna(row.max_i_ka) and row.max_i_ka > 0:
                # Use line-to-line voltage for 3-phase apparent power (this part was correct)
                v_ll_kv = self.bus_df.loc[row.from_bus, 'vn_kv'] 
                s_max_mva_thermal = math.sqrt(3) * v_ll_kv * row.max_i_ka
                s_max_pu = s_max_mva_thermal / (self.S_base_VA / 1e6) 
                self.bigM[l_idx] = s_max_pu 
            else:
                max_total_system_load_pu = sum(abs(p) for p in self.bus_p_inj.values()) 
                self.bigM[l_idx] = max_total_system_load_pu if max_total_system_load_pu > 0 else 10.0
        self.logger.info(f"bigM values: max {max(self.bigM.values()):.3f}, min {min(self.bigM.values()):.3f}")
        self.logger.info("Initialisation finished (%d buses, %d lines)",len(self.bus_df), len(self.line_df))

        if self.debug_level >= 2:
            self.logger.debug(f"Processed bus table: {len(self.bus_df)} buses")
            self.logger.debug(f"Processed line table: {len(self.line_df)} lines")
            self.logger.debug(f"Identified {len(self.lines_with_sw)} lines with switches")
            self.logger.debug(f"Identified {len(self.lines_wo_sw)} lines without switches")
            self.logger.debug(f"Big-M values: min={min(self.bigM.values()):.3f}, max={max(self.bigM.values()):.3f}")

        self.debug_network_summary()    

    def debug_network_summary(self):
        """Create a comprehensive summary of network properties for debugging"""
        if self.debug_level < 1:
            return
        
        summary = {
            "network_structure": {
                "buses": len(self.bus_df),
                "lines": len(self.line_df),
                "substations": len(self.substations),
                "active_buses": self.active_bus.sum(),
                "lines_with_switches": len(self.lines_with_sw),
                "lines_without_switches": len(self.lines_wo_sw),
                "total_switches": len(self.switch_df)
            },
            "network_parameters": {
                "voltage_base_min": min(self.v_base_V.values()),
                "voltage_base_max": max(self.v_base_V.values()),
                "S_base_VA": self.S_base_VA,
                "bigM_min": min(self.bigM.values()),
                "bigM_max": max(self.bigM.values()),
                "total_p_injection": sum(self.bus_p_inj.values()),
                "total_q_injection": sum(self.bus_q_inj.values()),
                "substation_buses": list(self.substations)
            }
        }
        
        # Add per-substation injection info
        summary["substation_injections"] = {}
        for sub in self.substations:
            summary["substation_injections"][sub] = {
                "p_inj": self.bus_p_inj.get(sub, 0),
                "q_inj": self.bus_q_inj.get(sub, 0),
                "connected_lines": len([l for l in self.line_df.index if 
                                       self.line_df.loc[l, "from_bus"] == sub or 
                                       self.line_df.loc[l, "to_bus"] == sub])
            }
        
        # Log key summary information
        self.logger.info("Network summary:")
        self.logger.info(f"  Buses: {summary['network_structure']['buses']} (active: {summary['network_structure']['active_buses']})")
        self.logger.info(f"  Lines: {summary['network_structure']['lines']}")
        self.logger.info(f"  Switches: {summary['network_structure']['total_switches']}")
        self.logger.info(f"  Substations: {summary['network_structure']['substations']}")
        self.logger.info(f"  Total P injection: {summary['network_parameters']['total_p_injection']:.4f} p.u.")
        self.logger.info(f"  Total Q injection: {summary['network_parameters']['total_q_injection']:.4f} p.u.")
        
        # Store for debugging
        self.debug_logs["network_summary"] = summary

    # ------------------------------------------------------------------------
    # 2.  CREATE-MODEL   
    # ------------------------------------------------------------------------
    def create_model(self) -> ConcreteModel:
        if self.bus_df is None:
            self.initialize()
       
        model = ConcreteModel()
        # -------------------- index sets
        active_buses = list(self.bus_df.index[self.active_bus])
        active_lines = [l for l in self.line_df.index
                        if self.active_bus[self.line_df.from_bus[l]]
                        and self.active_bus[self.line_df.to_bus[l]]]
        
        self.logger.info(f"Active buses: {len(active_buses)} of {len(self.bus_df)}")
        self.logger.info(f"Active lines: {len(active_lines)} of {len(self.line_df)}")

        model.buses = Set(initialize=active_buses, ordered=True)
        model.lines = Set(initialize=active_lines, ordered=True)
        model.times = Set(initialize=[0])
        model.partitionedbuses = Set(initialize=[b for b in active_buses if b not in self.substations])

        self.logger.info(f"Partitioned buses: {len(model.partitionedbuses)} of {len(model.buses)}")

        # -------------------- parameters
        v_upper_pu, v_lower_pu = 1.10, 0.90
        model.line_resistance_pu = Param(model.lines, initialize=lambda m, l: self.line_df.r_pu[l])
        model.line_reactance_pu = Param(model.lines, initialize=lambda m, l: self.line_df.x_pu[l])
        model.big_M_flow = Param(model.lines, initialize=lambda m, l: self.bigM[l] * self.bigM_factor)
        model.voltage_upper_bound = Param(model.buses, initialize=lambda m, b: v_upper_pu ** 2)
        model.voltage_lower_bound = Param(model.buses, initialize=lambda m, b: v_lower_pu ** 2)
        #model.big_M_flow =Param(model.lines, initialize=lambda m, l: max(self.bigM) * self.bigM_factor)
        self.logger.info(f"Big-M values: {self.bigM}")
        def calculate_voltage_bigM():
            v_upper_pu_squared = v_upper_pu ** 2
            v_lower_pu_squared = v_lower_pu ** 2
            max_voltage_deviation = v_upper_pu_squared - v_lower_pu_squared
            return max_voltage_deviation *self.voltage_bigM_factor
        voltage_bigM = calculate_voltage_bigM()
        model.voltage_bigM = voltage_bigM
        self.logger.info(f"Using voltage Big-M value: {voltage_bigM}")

        from_bus_map = {l: int(self.line_df.from_bus[l]) for l in model.lines}
        to_bus_map   = {l: int(self.line_df.to_bus[l])   for l in model.lines}
        model.from_bus = Param(model.lines, initialize=lambda m, l: from_bus_map[l])
        model.to_bus   = Param(model.lines, initialize=lambda m, l: to_bus_map[l])
        
        #DTDForbidden
        # :TODO remove big_M voltage add  slack
        # ------------------------------------------------------------------
        #  Variables
        # ------------------------------------------------------------------
        
        model.voltage_squared = Var(model.buses, model.times, within=NonNegativeReals, initialize=1.0)
        model.active_power_flow = Var(model.lines, model.times,within=Reals, initialize=0)
        model.reactive_power_flow = Var(model.lines, model.times,within=Reals, initialize=0)
        model.squared_current_magnitude = Var(model.lines, model.times, within=NonNegativeReals, initialize=0)
        
        model.bus_energised = Var(model.partitionedbuses, within=Binary, initialize=1)
        if not self.toggles.get("allow_load_shed", False):
            self.logger.info("Load shedding is not allowed. Fixing bus_energised variables to 1.")	
            for b in model.partitionedbuses:
                 model.bus_energised[b].fix(1)

        # Expression to get energization status 
        def is_up_rule(m, b):
            return m.bus_energised[b] if b in m.partitionedbuses else 1
        model.is_up = Expression(model.buses, rule=is_up_rule)
        
        # --------------------------------------------
        # Initialize slack variables
        # ---------------------------------------------

        model.voltage_drop_slack    = Var(model.lines,  model.times,within=NonNegativeReals, initialize=0)
        model.voltage_upper_slack = Var(model.buses, model.times, within=NonNegativeReals, initialize=0)
        model.voltage_lower_slack = Var(model.buses, model.times, within=NonNegativeReals, initialize=0)

        # ---------------------------------------------
        # Initialize line status variables
        # ---------------------------------------------
        model.line_status = Var(model.lines, within=Binary, initialize=lambda m, l: self.initial_line_status.get(l, 1))

        if self.toggles.get('all_lines_are_switches', False):
            model.line_initial_status = Param(model.lines, within=Binary, mutable=False,initialize=lambda m, l: self.initial_line_status.get(l, 1))
        else:
            for l in self.lines_wo_sw:
                model.line_status[l].fix(1)
            controllable_switches = list(self.switch_df[self.switch_df.et == "l"].index)
            model.switches = Set(initialize=controllable_switches)
            
            self.switches_for_line = self.line_switches["element"].to_dict()
            model_switch_indices = [s for s in model.switches if self.switches_for_line.get(s) in model.lines]
            model.model_switches = Set(initialize=model_switch_indices) 

            model.switch_status = Var(model.model_switches, within=Binary,initialize=lambda m, s: 1 if self.initial_switch_status.get(s, False) else 0)
            model.switch_initial = Param(model.model_switches, within=Binary, mutable=False,initialize=lambda m, s: 1 if self.initial_switch_status.get(s, False) else 0)
        
        if self.debug_level >= 1:
            try: 
                self.logger.debug(f"Initial line status: {self.initial_line_status}")
                self.logger.debug(f"Initial switch status: {self.initial_switch_status}")
            except:
                self.logger.warning("Error printing initial line/switch status. Check if they are set correctly.")

        # Add expression variables to track the flow at each bus for easy debugging
        model.sum_inflow = Expression(model.buses, rule=lambda m, b: sum(m.active_power_flow[l, 0] for l in m.lines if m.to_bus[l] == b))
        model.sum_outflow = Expression(model.buses,rule=lambda m, b: sum(m.active_power_flow[l, 0] for l in m.lines if m.from_bus[l] == b))
        model.flow_balance = Expression(model.buses, rule=lambda m, b: m.sum_outflow[b] - m.sum_inflow[b])
        
        self.enabled_constraints = {name: False for name in [
            "VoltageDropUp", "VoltageDropLo", "ActivePowerBalance", "ReactivePowerBalance",
            "VoltageUpper", "VoltageLower", "PFlowUB", "PFlowLB", "QFlowUB", "QFlowLB",
            "LineBusLinkFrom", "LineBusLinkTo", "BusConnectivity", "SpanConstraint",
            "RootCap", "RootBalance"
        ]}
        # ----------------------------------------------
        # Constraints
        # ----------------------------------------------

        def voltage_drop_upper(m, l, t):
            i, j = m.from_bus[l], m.to_bus[l]
            R, X = m.line_resistance_pu[l], m.line_reactance_pu[l]
            Mv   = m.voltage_bigM
            return (m.voltage_squared[j, t]- m.voltage_squared[i, t]+ 2*(R*m.active_power_flow[l, t] + X*m.reactive_power_flow[l, t])
                - (R**2+X**2)*m.squared_current_magnitude[l, t] 
                +m.voltage_drop_slack[l,t]
                <= (1 - m.line_status[l]) * Mv
            )
        def voltage_drop_lower(m, l, t):
            i, j = m.from_bus[l], m.to_bus[l]
            R, X = m.line_resistance_pu[l], m.line_reactance_pu[l]
            Mv   = m.voltage_bigM
            return (m.voltage_squared[j, t]- m.voltage_squared[i, t]+ 2*(R*m.active_power_flow[l, t] + X*m.reactive_power_flow[l, t])
                - (R**2+X**2)*m.squared_current_magnitude[l, t]
                - m.voltage_drop_slack[l,t]
                >= -(1 - m.line_status[l]) * Mv)

        if self.toggles.get("include_voltage_drop_constraint", True):
            model.VoltageDropUp = Constraint(model.lines, model.times, rule=voltage_drop_upper)
            self.enabled_constraints["VoltageDropUp"] = True
            model.VoltageDropLo = Constraint(model.lines, model.times, rule=voltage_drop_lower)
            self.enabled_constraints["VoltageDropLo"] = True

        def active_power_balance(m, b, t):
            if b in self.substations:
                return Constraint.Skip
            out_q = sum(m.active_power_flow[l, t] for l in m.lines if m.from_bus[l] == b)
            in_q  = sum(m.active_power_flow[l, t] for l in m.lines if m.to_bus[l]   == b)
            injection = self.bus_p_inj[b] * m.is_up[b]
            return out_q - in_q ==injection * m.is_up[b] 
        
        def reactive_power_balance(m, b, t):
            if b in self.substations: 
                return Constraint.Skip
            out_q = sum(m.reactive_power_flow[l, t] for l in m.lines if m.from_bus[l] == b)
            in_q  = sum(m.reactive_power_flow[l, t] for l in m.lines if m.to_bus[l]   == b)
            return out_q - in_q == self.bus_q_inj[b] * m.is_up[b] 

        if self.toggles["include_power_balance_constraint"]:
            model.ActivePowerBalance = Constraint(model.buses, model.times, rule=active_power_balance)
            model.ReactivePowerBalance = Constraint(model.buses, model.times, rule=reactive_power_balance)
            self.enabled_constraints["ActivePowerBalance"] = True
            self.enabled_constraints["ReactivePowerBalance"] = True

        # def link_switch_line(m, l):
        #     s = self.line_to_switch[l]
        #     return m.line_status[l] == m.switch_status[s]
        # model.LinkSwitch = Constraint(model.lines, rule=link_switch_line)
        def current_line_coupling(m, l, t):
            return m.squared_current_magnitude[l, t] <= m.big_M_flow[l]**2 * m.line_status[l]
        model.CurrentLineCoupling = Constraint(model.lines, model.times, rule=current_line_coupling)


        # (29) second-order cone: ‖(2P,2Q,V_i-ℓ)‖₂ ≤ V_i+ℓ
        def socp_cone(m, l, t):
            i = m.from_bus[l]
            
            #lhs = (2 * m.active_power_flow[l, t])**2 + (m.reactive_power_flow[l, t])**2
            lhs = (m.active_power_flow[l, t])**2 + (m.reactive_power_flow[l, t])**2
            
            # Right-hand side: V_i * I_ij
            rhs = m.voltage_squared[i, t] * m.squared_current_magnitude[l, t]
            return lhs <= rhs + (1 - m.line_status[l]) * m.big_M_flow[l]**2

        if self.toggles["include_cone_constraint"]:
            model.SOC_Cone = Constraint(model.lines, model.times, rule=socp_cone)
            self.enabled_constraints["SOC_Cone"] = True

            def P_flow_ub(m, l, t):
                return  m.active_power_flow[l, t] <=  m.big_M_flow[l] * m.line_status[l]
            def P_flow_lb(m, l, t):
                return -m.active_power_flow[l, t] <=  m.big_M_flow[l] * m.line_status[l]
            def Q_flow_ub(m, l, t):
                return  m.reactive_power_flow[l, t] <=  m.big_M_flow[l] * m.line_status[l]
            def Q_flow_lb(m, l, t):
                return -m.reactive_power_flow[l, t] <=  m.big_M_flow[l] * m.line_status[l]
            model.PFlowUB = Constraint(model.lines, model.times, rule=P_flow_ub)
            model.PFlowLB = Constraint(model.lines, model.times, rule=P_flow_lb)
            model.QFlowUB = Constraint(model.lines, model.times, rule=Q_flow_ub)
            model.QFlowLB = Constraint(model.lines, model.times, rule=Q_flow_lb)
            self.enabled_constraints["PFlowUB"] = True
            self.enabled_constraints["PFlowLB"] = True
            self.enabled_constraints["QFlowUB"] = True
            self.enabled_constraints["QFlowLB"] = True

        #voltag_upper_slack = v

        # -------- voltage bounds   ----------------------
        # (30)-(31) Voltage bounds with big‑M relaxation
        # def voltage_upper(m, b, t):
        #     return m.voltage_squared[b, t] <= m.voltage_upper_bound[b]

        # def voltage_lower(m, b, t):
        #     return m.voltage_squared[b, t] >= m.voltage_lower_bound[b] 

        def voltage_upper(m, b, t):
            return m.voltage_squared[b,t] <= m.voltage_upper_bound[b] + m.voltage_upper_slack[b,t]
        def voltage_lower(m, b, t):
            return m.voltage_squared[b,t] >= m.voltage_lower_bound[b] - m.voltage_lower_slack[b,t]
    
        
        if self.toggles["include_voltage_bounds_constraint"]:
            model.VoltageUpper = Constraint(model.buses, model.times, rule=voltage_upper)
            model.VoltageLower = Constraint(model.buses, model.times, rule=voltage_lower)
            self.enabled_constraints["VoltageUpper"] = True
            self.enabled_constraints["VoltageLower"] = True

        # --- Line Status Linking Constraints ---
        # 1. Line active only if connected buses are energized
        def line_bus_link_from(m, l):
             return m.line_status[l] <= m.is_up[m.from_bus[l]]
        model.LineBusLinkFrom = Constraint(model.lines, rule=line_bus_link_from)
        self.enabled_constraints["LineBusLinkFrom"] = True
        def line_bus_link_to(m, l):
             return m.line_status[l] <= m.is_up[m.to_bus[l]]
        model.LineBusLinkTo = Constraint(model.lines, rule=line_bus_link_to)
        self.enabled_constraints["LineBusLinkTo"] = True

        # --- Radiality Constraints ---
        if self.toggles["include_radiality_constraints"]:
            def bus_connectivity_rule(m, b):
                 if b in m.partitionedbuses: 
                     inc = [l_idx for l_idx in m.lines if m.from_bus[l_idx] == b or m.to_bus[l_idx] == b]
                     if not inc:
                         self.logger.warning(f"BusConnectivity for Bus {b}: No incident lines. Constraint implies 0 >= 1.")
                         return Constraint.Skip # 0 == m.is_up[b] 
                     return sum(m.line_status[l_idx] for l_idx in inc) >= m.is_up[b]
                 return Constraint.Skip
            model.BusConnectivity = Constraint(model.partitionedbuses, rule=bus_connectivity_rule)
            self.enabled_constraints["BusConnectivity"] = True
             
            
            if self.toggles["use_root_flow"]:
                self.logger.info("Using single commodity flow for radiality constraints")
                model.arcs = Set(initialize=[(i, j, l) for l in model.lines
                        for (i,j) in ((model.from_bus[l], model.to_bus[l]),
                                      (model.to_bus[l],   model.from_bus[l]))], dimen=3)
                # --- SCF Flow Variable ---
                # scf_flow_var[i,j,l] is the commodity flow from bus i to bus j over line l
                model.scf_flow_var = Var(model.arcs, within=NonNegativeReals, initialize=0)

                # --- Designate a Single Root for Commodity Flow ---
                active_model_substations = [s for s in self.substations if s in model.buses]
                
                if not active_model_substations:
                    if not list(model.buses): # Check if model.buses is empty
                        self.logger.error("SCF Radiality: No buses in the model. Cannot designate a root.")
                        designated_root = None 
                    else:
                        designated_root = model.buses.first() 
                        self.logger.warning(f"SCF Radiality: No active substations in the model. Designated {designated_root} as root.")
                else:
                    designated_root = active_model_substations[0]
                    self.logger.info(f"SCF Radiality: Designated root for commodity flow is {designated_root}")


                    # Pre-calculate maximum possible downstream loads for each arc
                    arc_max_flows = self._calculate_arc_max_flows(model, designated_root)
                    M_scf = len(model.buses) - 1 if len(model.buses) > 0 else 0
                    # --- Flow Capacity Constraints ---
                    
                    self.logger.info("Using single commodity flow for radiality constraints")
                
                    # --- Designate a Single Root for Commodity Flow ---
                    active_model_substations = [s for s in self.substations if s in model.buses]
                    
                    if not active_model_substations:
                        if not list(model.buses): # Check if model.buses is empty
                            self.logger.error("SCF Radiality: No buses in the model. Cannot designate a root.")
                            # This is a critical error, model building should probably stop or handle this.
                            # For now, let's assume this won't happen if preprocessing is correct.
                            designated_root = None 
                        else:
                            designated_root = model.buses.first() # Pick the first bus in the set if no substations
                            self.logger.warning(f"SCF Radiality: No active substations in the model. Designated {designated_root} as root.")
                    else:
                        designated_root = active_model_substations[0]
                        self.logger.info(f"SCF Radiality: Designated root for commodity flow is {designated_root}")

                    if designated_root is not None:
                        # --- Flow Capacity Constraint ---
                        # Flow on an arc is only possible if the underlying line is active (line_status[l]=1)
                        # Max flow on an arc is N_buses (or N_energized_buses - 1)
                        # (ensures flow variable goes to zero if line is off)
                        n_buses_total = len(model.buses) # Max possible demand
                        def scf_capacity_rule(m, u, v, l):
                            return m.scf_flow_var[u,v,l] <= (n_buses_total -1) * m.line_status[l]
                        model.SCFCapacity = Constraint(model.arcs, rule=scf_capacity_rule)
                        self.enabled_constraints["SCFCapacity"] = True

                        # --- Flow Conservation Constraint ---
                        def scf_flow_conservation_rule(m, b):
                            # Sum of commodity flows into bus b
                            inflow = sum(m.scf_flow_var[u,b_val,l_idx] for (u,b_val,l_idx) in m.arcs if b_val == b)
                            # Sum of commodity flows out of bus b
                            outflow = sum(m.scf_flow_var[b_val,v,l_idx] for (b_val,v,l_idx) in m.arcs if b_val == b)

                            if b == designated_root:
                                # The designated root supplies 1 unit of commodity for each *other* energized bus.
                                # These other energized buses become the "sinks" in the commodity flow model.
                                num_other_energized_sinks = sum(m.is_up[n_bus] for n_bus in m.buses if n_bus != designated_root)
                                return outflow - inflow == num_other_energized_sinks
                            else:
                                # Any other bus (non-root substations or partitioned buses) demands 1 unit of commodity if energized.
                                return inflow - outflow == 1 * m.is_up[b]
                        
                        model.SCFFlowConservation = Constraint(model.buses, rule=scf_flow_conservation_rule)
                    # --- Spanning Tree Constraint with Load Shedding Support ---
                    def spanning_tree_rule(m):
                        total_energized_buses = sum(m.is_up[b] for b in m.buses)
                        return sum(m.line_status[l] for l in m.lines) == total_energized_buses - 1
                    
                    model.SpanningTree = Constraint(rule=spanning_tree_rule)
                

                    # --- Bus Energization Logic ---
                    if not self.toggles.get("allow_load_shed", False):
                        # Force energization of all reachable buses
                        def bus_forced_energization_rule(m, b):
                            if b in m.partitionedbuses:
                                return m.is_up[b] == 1
                            return Constraint.Skip
                        
                        model.SCFBusForcedEnergization = Constraint(model.buses, rule=bus_forced_energization_rule)
                        self.enabled_constraints["SCFBusForcedEnergization"] = True
        
        # -------------------- objective (25) + penalties
        # --- Objective Term Expressions ---
        def loss_term_rule(m):
            return sum(m.line_resistance_pu[l] * m.squared_current_magnitude[l, 0]
                for l in m.lines)
        model.loss_term_expr = Expression(rule=loss_term_rule)

        def voltage_drop_slack_term_rule(m):
            return self.slack_penalty * sum(m.voltage_drop_slack[l, 0]**2 for l in m.lines)   
        model.voltage_drop_slack_term_rule = Expression(rule=voltage_drop_slack_term_rule)

        
        # and add into objective something like:
        def voltage_bounds_slack_term_rule(m):
            return self.voltage_slack_penalty * (
                sum(m.voltage_upper_slack[b,0] + m.voltage_lower_slack[b,0] for b in m.buses)
            )
        model.voltage_bounds_slack_term_rule = Expression(rule=voltage_bounds_slack_term_rule)
      
         # Default if not included
        if self.toggles["include_switch_penalty"]:
            if self.toggles.get('all_lines_are_switches', False):
                def switch_term_rule(m):
                    return self.switch_penalty * sum(
                        m.line_initial_status[l] * (1 - m.line_status[l]) +
                        (1 - m.line_initial_status[l]) * m.line_status[l]
                        for l in m.lines
                    )
            else:
                def switch_term_rule(m):
                     if not hasattr(m,'model_switches'): return 0.0
                     return self.switch_penalty * sum(
                         m.switch_initial[s] * (1 - m.switch_status[s]) +
                         (1 - m.switch_initial[s]) * m.switch_status[s]
                         for s in m.model_switches)
            model.switch_pen_expr = Expression(rule=switch_term_rule)
        else:
            model.switch_pen_expr = Expression(initialize=0.0)

        def load_shed_costs(m):
            # Load shedding penalty
                return self.load_shed_penalty * sum(
                    abs(self.bus_p_inj[b]) * (1 - m.is_up[b]) 
                    for b in m.partitionedbuses if self.bus_p_inj.get(b, 0) < 0 )
        if self.toggles.get("allow_load_shed", False):
            model.load_shed_cost = Expression(rule=load_shed_costs)
            self.logger.info("Load shedding costs enabled with penalty %.2f", self.load_shed_penalty)

            
        # --- Combine Terms for Objective ---
        def objective_rule(m):
            load_shed_cost = 0
            if self.toggles.get("allow_load_shed", False):
                load_shed_cost = m.load_shed_cost
            return m.loss_term_expr + m.voltage_drop_slack_term_rule + m.switch_pen_expr  +load_shed_cost + m.voltage_bounds_slack_term_rule
        model.objective = Objective(rule=objective_rule, sense=minimize)
        


        self.model = model
        if self.debug_level >= 1:
            self.logger.debug("\n%s", build_model_size_report(model))
        self.validate_network_connectivity()
        return model
    

    def solve(self, *, solver: str = "gurobi_persistent",
              time_limit: int = 6000, threads: int = 8,
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
            if self.debug_level >= 2:
                opt.options.update({"Presolve": 0, "OutputFlag": 1})
        
        # user overrides
        opt.options.update(solver_kw)

        self.solver_results = opt.solve(tee=self.debug_level >1, load_solutions=True)
        self.solve_time = time.time() - start

        obj_val = pyo_val(self.model.objective)
        self.logger.info("Solved in %.2fs  (obj = %.3f)",
                self.solve_time, obj_val)
        
        if self.debug_level >= 1:
            termination_condition = self.solver_results.solver.termination_condition
            self.logger.info(f"Solver termination condition: {termination_condition}")
            
            if str(termination_condition) != "optimal":
                self.logger.warning("Solution is not optimal.")
                if self.debug_level >= 2:
                    self.logger.debug("Running infeasibility diagnosis...")
                    from pyomo.util.infeasible import log_infeasible_constraints
                    log_infeasible_constraints(self.model, log_expression=True, log_variables=True)
    

        # Deep verification of solution 
        self.verify_solution()
        
        # Verify constraint satisfaction
        self.verify_constraint_satisfaction()
        
        # Track switch changes
        self.track_switch_changes()
        
        # Process the solution for return
        self.process_solution(update_network=False)
        return self.solver_results

    def track_switch_changes(self):
        """Track and analyze the changes in switch status from initial conditions."""
        m = self.model
        
        switch_change_count = 0
        total_switches = 0
        
        # Count energized buses
        energized_count = sum(1 for b in m.buses if round(pyo_val(m.is_up[b])) == 1)
        total_buses = len(m.buses)
        self.logger.info(f"Energized buses: {energized_count} out of {total_buses}")

        # List de-energized buses if any
        if energized_count < total_buses:
            de_energized = [b for b in m.buses if round(pyo_val(m.is_up[b])) == 0]
            self.logger.warning(f"De-energized buses: {de_energized}")
        
        # Track switch changes based on model configuration
        switch_changes = []
        
        if not self.toggles.get('all_lines_are_switches', False) and hasattr(m, 'model_switches'):
            # Using explicit switch model
            self.logger.info("Using explicit switch model - checking switch status changes:")
            
            try:
                for s_idx in m.model_switches:
                    original_status = bool(pyo_val(m.switch_initial[s_idx]))
                    optimized_status = bool(round(pyo_val(m.switch_status[s_idx])))
                    
                    status_changed = original_status != optimized_status
                    if status_changed:
                        switch_change_count += 1
                        switch_changes.append({
                            'switch_id': s_idx,
                            'original': original_status,
                            'optimized': optimized_status
                        })
                    
                    if status_changed or self.debug_level >= 2:
                        # Print all changes, or all switches in verbose debug mode
                        self.logger.info(f"  Switch {s_idx}: {original_status} → {optimized_status}" + 
                                        (" (CHANGED)" if status_changed else ""))
                    
                    total_switches += 1
            except Exception as e:
                self.logger.error(f"Error comparing switch statuses: {e}")
        
        else:
            # Using all_lines_are_switches model
            self.logger.info("Using all_lines_are_switches model - checking line status changes:")
            
            try:
                for l_idx in m.lines:
                    if l_idx in self.lines_with_sw:  # Only check lines with switches
                        original_status = bool(pyo_val(m.line_initial_status[l_idx]))
                        optimized_status = bool(round(pyo_val(m.line_status[l_idx])))
                        
                        status_changed = original_status != optimized_status
                        if status_changed:
                            switch_change_count += 1
                            switch_changes.append({
                                'line_id': l_idx,
                                'original': original_status,
                                'optimized': optimized_status
                            })
                            
                        if status_changed or self.debug_level >= 2:
                            # Print all changes, or all switchable lines in verbose debug mode
                            self.logger.info(f"  Line {l_idx}: {original_status} → {optimized_status}" + 
                                            (" (CHANGED)" if status_changed else ""))
                        
                        total_switches += 1
            except Exception as e:
                self.logger.error(f"Error comparing line statuses: {e}")
        
        # Summary 
        self.logger.info(f"Total switches/switchable lines: {total_switches}")
        percentage = (switch_change_count/total_switches*100) if total_switches > 0 else 0
        self.logger.info(f"Switches/lines changed by optimization: {switch_change_count} ({percentage:.1f}%)")
        
        # Store for later reference
        self.num_switches_changed = switch_change_count
        self.debug_logs["switch_changes"] = switch_changes
        
        # Detect potential issues with the solution
        if switch_change_count == 0 and total_switches > 0:
            self.logger.warning("WARNING: NO SWITCHES CHANGED IN OPTIMIZATION SOLUTION!")
            
            # Additional diagnostics for radiality
            if self.toggles.get("use_spanning_tree_radiality", False) and hasattr(m, "SpanConstraint"):
                active_lines = sum(1 for l in m.lines if round(pyo_val(m.line_status[l])) == 1)
                expected_active = len(m.buses) - 1
                lhs = sum(pyo_val(m.line_status[l]) for l in m.lines)
                rhs = len(m.buses) - 1
                self.logger.warning(f"Spanning tree check: Active lines = {active_lines}, Expected = {expected_active}")
                self.logger.warning(f"SpanConstraint: sum(line_status) = {lhs} == {rhs} ?: {abs(lhs-rhs) < 1e-6}")
        else:
            self.logger.info(f"SUCCESS: Optimization changed {switch_change_count} switches")
    
    def verify_solution(self):
        """Perform deep verification of the solution to identify issues."""
        if not self.model:
            self.logger.warning("No model exists. Call create_model() and solve() first.")
            return
            
        m = self.model
        
        # Store key variable values for debugging
        self.variable_values = {}
        
        # 1. Voltage squared values
        v_squared = {}
        for b in m.buses:
            for t in m.times:
                v_squared[(b, t)] = pyo_val(m.voltage_squared[b, t])
        self.variable_values["voltage_squared"] = v_squared
        
        # 2. Power flows
        p_flows = {}
        q_flows = {}
        for l in m.lines:
            for t in m.times:
                p_flows[(l, t)] = pyo_val(m.active_power_flow[l, t])
                q_flows[(l, t)] = pyo_val(m.reactive_power_flow[l, t])
        self.variable_values["p_flows"] = p_flows
        self.variable_values["q_flows"] = q_flows
        
        # 3. Line status
        line_status = {}
        for l in m.lines:
            line_status[l] = round(pyo_val(m.line_status[l]))
        self.variable_values["line_status"] = line_status
        
        # 4. Slack variables if used
        if self.toggles.get("use_slack_variables", False):
            p_slack = {}
            q_slack = {}
            v_slack = {}
            
            for b in m.buses:
                for t in m.times:
                    p_slack[(b, t)] = pyo_val(m.active_power_slack[b, t])
                    q_slack[(b, t)] = pyo_val(m.reactive_power_slack[b, t])
            
            for l in m.lines:
                for t in m.times:
                    v_slack[(l, t)] = pyo_val(m.voltage_drop_slack[l, t])
                    
            self.variable_values["p_slack"] = p_slack
            self.variable_values["q_slack"] = q_slack
            self.variable_values["v_slack"] = v_slack
            
            # Report on significant slack values
            significant_p_slack = [(k, v) for k, v in p_slack.items() if abs(v) > 1e-4]
            significant_q_slack = [(k, v) for k, v in q_slack.items() if abs(v) > 1e-4]
            significant_v_slack = [(k, v) for k, v in v_slack.items() if abs(v) > 1e-4]
            
            if significant_p_slack:
                self.logger.warning(f"Significant active power slack values detected: {len(significant_p_slack)}")
                for (b, t), val in significant_p_slack[:5]:  # Show first 5
                    self.logger.warning(f"  Bus {b}, t={t}: P-slack = {val:.6g}")
            
            if significant_q_slack:
                self.logger.warning(f"Significant reactive power slack values detected: {len(significant_q_slack)}")
                for (b, t), val in significant_q_slack[:5]:  # Show first 5
                    self.logger.warning(f"  Bus {b}, t={t}: Q-slack = {val:.6g}")
            
            if significant_v_slack:
                self.logger.warning(f"Significant voltage drop slack values detected: {len(significant_v_slack)}")
                for (l, t), val in significant_v_slack[:5]:  # Show first 5
                    self.logger.warning(f"  Line {l}, t={t}: V-slack = {val:.6g}")
        
        # 5. Check objective components
    
        self.logger.info("Objective function breakdown:")
        self.logger.info(f"  Total objective value: { pyo_val(m.objective):.6g}")
        self.logger.info(f"  Loss term: {pyo_val(m.loss_term_expr):.6g}")
        self.logger.info(f"  Voltage drop slack term: {pyo_val(m.voltage_drop_slack_term_rule) :.6g}")
        self.logger.info(f"  Voltage bounds slack term: {pyo_val(m.voltage_bounds_slack_term_rule):.6g}")
        self.logger.info(f"  Switch penalty term: {pyo_val(m.switch_pen_expr):.6g}")
        
        # 6. Verify network structure
        active_lines = sum(1 for l in m.lines if line_status[l] > 0.5)
        self.logger.info(f"Active lines in solution: {active_lines} of {len(m.lines)}")
        
        # 7. Check radiality condition
        if self.toggles.get("use_spanning_tree_radiality", False):
            nbuses = len(m.buses)
            self.logger.info(f"Spanning tree check: {active_lines} active lines, {nbuses-1} expected for a tree")
            
            if active_lines != nbuses - 1:
                self.logger.warning("Solution might not form a proper spanning tree!")
    
    def verify_constraint_satisfaction(self):
        """Verify if all model constraints are satisfied by the solution."""
        if not self.model:
            self.logger.warning("No model exists. Call create_model() and solve() first.")
            return
            
        m = self.model
        tolerance = 1e-5
        
        # Store constraint violations
        self.constraint_violations = {}
        
        # Check all constraints in the model
        for c in m.component_data_objects(Constraint, active=True):
            try:
                # Get constraint bounds
                lb = c.lower if c.has_lb() else None
                ub = c.upper if c.has_ub() else None
                
                # Calculate constraint body value
                body_val = pyo_val(c.body)
                
                # Check for violation
                if (lb is not None and body_val < lb - tolerance) or (ub is not None and body_val > ub + tolerance):
                    # Record violation
                    violation = {
                        "constraint_name": c.name,
                        "body_value": body_val,
                        "lower_bound": lb,
                        "upper_bound": ub,
                        "violation": max(lb - body_val if lb is not None else 0, 
                                       body_val - ub if ub is not None else 0)
                    }
                    
                    # Group violations by constraint type
                    constraint_type = c.name.split('[')[0]
                    if constraint_type not in self.constraint_violations:
                        self.constraint_violations[constraint_type] = []
                    
                    self.constraint_violations[constraint_type].append(violation)
                    
                    # Log major violations
                    if violation["violation"] > 1e-3:
                        self.logger.warning(f"Constraint violation: {c.name} = {body_val:.6g}")
                        if lb is not None and body_val < lb - tolerance:
                            self.logger.warning(f"  Value {body_val:.6g} < {lb} (lower bound) by {lb - body_val:.6g}")
                        if ub is not None and body_val > ub + tolerance:
                            self.logger.warning(f"  Value {body_val:.6g} > {ub} (upper bound) by {body_val - ub:.6g}")
                        
            except Exception as e:
                self.logger.error(f"Error checking constraint {c.name}: {e}")
        
        # Summarize violations
        total_violations = sum(len(violations) for violations in self.constraint_violations.values())
        if total_violations > 0:
            self.logger.warning(f"Found {total_violations} constraint violations")
            
            # Show violations by constraint type
            for constraint_type, violations in self.constraint_violations.items():
                worst_violation = max(violations, key=lambda v: v["violation"])
                self.logger.warning(f"  {constraint_type}: {len(violations)} violations, worst: {worst_violation['violation']:.6g}")
        else:
            self.logger.info("All constraints satisfied within tolerance.")
        
        # Focus on key constraints if violations exist
        if "ActivePowerBalance" in self.constraint_violations or "ReactivePowerBalance" in self.constraint_violations:
            self.logger.warning("Power balance constraints violated. Checking power balance in detail...")
            self.verify_power_balance_constraints()

    def update_network(self):
        """Update the network with the optimization results."""
        net_updated = self.net.deepcopy()
        
        # Update switch statuses based on optimization results
        if hasattr(self.model, 'model_switches'):
            for s_idx in self.model.model_switches:
                # Get optimized status (0 or 1)
                switch_closed = bool(round(pyo_val(self.model.switch_status[s_idx])))
                # Apply to network
                if s_idx in net_updated.switch.index:
                    net_updated.switch.at[s_idx, 'closed'] = switch_closed
        elif self.toggles.get('all_lines_are_switches', False):
            # If using the all_lines_are_switches mode
            for l_idx in self.model.lines:
                if l_idx in self.lines_with_sw:
                    # Get line status
                    line_active = bool(round(pyo_val(self.model.line_status[l_idx])))
                    # Update all switches associated with this line
                    for s_idx in self.lines_with_sw[l_idx]:
                        if s_idx in net_updated.switch.index:
                            net_updated.switch.at[s_idx, 'closed'] = line_active
        
        # Log switch changes
        if self.debug_level >= 1:
            self.logger.debug("Network updated with optimization results")
            self.logger.debug(f"Total switches changed: {self.num_switches_changed}")
        
        return net_updated

    def _calculate_arc_max_flows(self, model, root_bus):
        """Pre-calculate maximum possible flow on each arc based on network topology"""
        import networkx as nx
        
        # Build a graph representation
        G = nx.Graph()
        for l in model.lines:
            from_bus = model.from_bus[l]
            to_bus = model.to_bus[l]
            G.add_edge(from_bus, to_bus, line_id=l)
        
        arc_max_flows = {}
        
        for u, v, l in model.arcs:
            # Calculate maximum possible downstream demand from this arc
            try:
                # Remove this edge temporarily to find downstream buses
                G_temp = G.copy()
                if G_temp.has_edge(u, v):
                    G_temp.remove_edge(u, v)
                
                # Find connected components
                components = list(nx.connected_components(G_temp))
                
                # Identify which component contains the root
                root_component = None
                downstream_component = None
                
                for comp in components:
                    if root_bus in comp:
                        root_component = comp
                    elif v in comp:  # The 'to' node of the arc
                        downstream_component = comp
                
                if downstream_component:
                    # Count buses in downstream component that are in partitioned buses
                    downstream_demand = len([b for b in downstream_component if b in model.partitionedbuses])
                    arc_max_flows[(u, v, l)] = max(1, downstream_demand)
                else:
                    # Default conservative estimate
                    arc_max_flows[(u, v, l)] = len(model.buses) // 2
                    
            except Exception as e:
                # Fallback to conservative estimate
                self.logger.warning(f"Could not calculate max flow for arc ({u},{v},{l}): {e}")
                arc_max_flows[(u, v, l)] = len(model.buses) - 1
        
        self.logger.debug(f"Calculated arc max flows: {len(arc_max_flows)} arcs processed")
        return arc_max_flows

    # Validate network connectivity for flow modeling
    def validate_network_connectivity(self):
        """Validate the network connectivity to ensure flow modeling is possible."""
        self.logger.info("Validating network connectivity for flow modeling...")
        
        # Check for isolated buses
        isolated_buses = []
        for b in self.model.buses:
            incident_lines = [l for l in self.model.lines 
                            if self.model.from_bus[l] == b or self.model.to_bus[l] == b]
            if not incident_lines:
                isolated_buses.append(b)
                self.logger.warning(f"Bus {b} is isolated with no incident lines!")
        
        if isolated_buses:
            self.logger.warning(f"Found {len(isolated_buses)} isolated buses: {isolated_buses}")
        else:
            self.logger.info("No isolated buses found!")
        
        # Verify substations have connections
        for b in self.substations:
            if b in self.model.buses:  # Only check active substations
                incident_lines = [l for l in self.model.lines 
                                if self.model.from_bus[l] == b or self.model.to_bus[l] == b]
                if not incident_lines:
                    self.logger.error(f"Substation bus {b} has no incident lines but is expected to inject power!")
                else:
                    self.logger.info(f"Substation bus {b} has {len(incident_lines)} incident lines.")
        
        # Check injection values
        zero_injection_buses = 0
        for b in self.model.buses:
            if b not in self.substations and abs(self.bus_p_inj[b]) < 1e-6:
                zero_injection_buses += 1
        
        self.logger.info(f"Found {zero_injection_buses} buses with approximately zero active power injection.")
        
        # Check radiality if spanning tree is used
        if self.toggles.get("use_spanning_tree_radiality", False):
            num_buses = len(self.model.buses)
            num_lines = len(self.model.lines)
            if num_lines != num_buses - 1:
                self.logger.warning(f"Network has {num_lines} lines and {num_buses} buses. " 
                                f"For a radial network, expected {num_buses - 1} lines.")
        
        # Store connectivity information for debugging
        self.debug_logs["network_connectivity"] = {
            "isolated_buses": isolated_buses,
            "zero_injection_buses": zero_injection_buses,
            "substation_connections": {
                b: len([l for l in self.model.lines 
                        if self.model.from_bus[l] == b or self.model.to_bus[l] == b])
                for b in self.substations if b in self.model.buses
            }
        }

    def verify_power_balance_constraints(self):
        """
        Detailed verification of power balance constraints to identify issues.
        """
        self.logger.info("Verifying power balance constraints...")
        
        # Track violations
        active_power_violations = []
        reactive_power_violations = []
        
        # Checking each bus for active power balance
        for b in self.model.buses:
            # Get flows, injections and balance values
            outflow = pyo_val(self.model.sum_outflow[b])
            inflow = pyo_val(self.model.sum_inflow[b])
            balance = outflow - inflow
            
            # For non-substation buses
            if b not in self.substations:
                injection = self.bus_p_inj[b] * pyo_val(self.model.is_up[b])
                p_slack = pyo_val(self.model.active_power_slack[b, 0]) if hasattr(self.model, "active_power_slack") else 0
                expected = injection + p_slack
                
                error = abs(balance - expected)
                if error > 1e-4:  # Tolerance for numerical issues
                    active_power_violations.append({
                        "bus": b,
                        "outflow": outflow,
                        "inflow": inflow,
                        "balance": balance,
                        "injection": injection, 
                        "slack": p_slack,
                        "expected": expected,
                        "error": error
                    })
            
            # For reactive power (skip substations)
            if b not in self.substations:
                # Calculate reactive flows
                q_outflow = sum(pyo_val(self.model.reactive_power_flow[l, 0]) 
                               for l in self.model.lines if self.model.from_bus[l] == b)
                q_inflow = sum(pyo_val(self.model.reactive_power_flow[l, 0]) 
                              for l in self.model.lines if self.model.to_bus[l] == b)
                q_balance = q_outflow - q_inflow
                
                q_injection = self.bus_q_inj[b] * pyo_val(self.model.is_up[b])
                q_slack = pyo_val(self.model.reactive_power_slack[b, 0]) if hasattr(self.model, "reactive_power_slack") else 0
                q_expected = q_injection + q_slack
                
                q_error = abs(q_balance - q_expected)
                if q_error > 1e-4:  # Tolerance for numerical issues
                    reactive_power_violations.append({
                        "bus": b,
                        "outflow": q_outflow,
                        "inflow": q_inflow,
                        "balance": q_balance,
                        "injection": q_injection,
                        "slack": q_slack,
                        "expected": q_expected,
                        "error": q_error
                    })
        
        # Log results
        if active_power_violations:
            worst_active = max(active_power_violations, key=lambda v: v["error"])
            self.logger.warning(f"Found {len(active_power_violations)} active power balance violations")
            self.logger.warning(f"Worst active power violation at bus {worst_active['bus']}: error = {worst_active['error']:.6g}")
            
            # Log details of first few violations
            for violation in active_power_violations[:5]:
                b = violation["bus"]
                self.logger.warning(f"  Bus {b}: outflow={violation['outflow']:.6g}, inflow={violation['inflow']:.6g}")
                self.logger.warning(f"    balance={violation['balance']:.6g}, injection={violation['injection']:.6g}")
                if violation["slack"] != 0:
                    self.logger.warning(f"    slack={violation['slack']:.6g}")
                self.logger.warning(f"    expected={violation['expected']:.6g}, error={violation['error']:.6g}")
        else:
            self.logger.info("No active power balance violations found!")
            
        if reactive_power_violations:
            worst_reactive = max(reactive_power_violations, key=lambda v: v["error"])
            self.logger.warning(f"Found {len(reactive_power_violations)} reactive power balance violations")
            self.logger.warning(f"Worst reactive power violation at bus {worst_reactive['bus']}: error = {worst_reactive['error']:.6g}")
            
            # Log details of first few violations
            for violation in reactive_power_violations[:5]:
                b = violation["bus"]
                self.logger.warning(f"  Bus {b}: outflow={violation['outflow']:.6g}, inflow={violation['inflow']:.6g}")
                self.logger.warning(f"    balance={violation['balance']:.6g}, injection={violation['injection']:.6g}")
                if violation["slack"] != 0:
                    self.logger.warning(f"    slack={violation['slack']:.6g}")
                self.logger.warning(f"    expected={violation['expected']:.6g}, error={violation['error']:.6g}")
        else:
            self.logger.info("No reactive power balance violations found!")
            
        # Store violations for debugging
        self.debug_logs["power_balance_violations"] = {
            "active": active_power_violations,
            "reactive": reactive_power_violations
        }
        
        # Check special case: zero-injection buses with non-zero flows
        zero_injection_with_flow = []
        for b in self.model.buses:
            if b not in self.substations and abs(self.bus_p_inj[b]) < 1e-6:
                flow_balance = pyo_val(self.model.flow_balance[b])
                if abs(flow_balance) > 1e-4:
                    zero_injection_with_flow.append({
                        "bus": b,
                        "p_injection": self.bus_p_inj[b],
                        "flow_balance": flow_balance
                    })
        
        if zero_injection_with_flow:
            self.logger.warning(f"Found {len(zero_injection_with_flow)} zero-injection buses with non-zero flow balance")
            for item in zero_injection_with_flow[:5]:
                self.logger.warning(f"  Bus {item['bus']}: p_injection={item['p_injection']:.6g}, flow_balance={item['flow_balance']:.6g}")
            
            self.debug_logs["zero_injection_flow_issues"] = zero_injection_with_flow
    
    def verify_voltage_drop_constraints(self):
        """
        Verify the voltage drop constraints for all active lines.
        This helps identify issues with the voltage drop formulation.
        """
        self.logger.info("Verifying voltage drop constraints...")
        
        # Track violations
        voltage_drop_violations = []
        
        # Check each line
        for l in self.model.lines:
            # Only check active lines
            if pyo_val(self.model.line_status[l]) < 0.5:
                continue
                
            # Get parameters
            i, j = self.model.from_bus[l], self.model.to_bus[l]
            R, X = self.model.line_resistance_pu[l], self.model.line_reactance_pu[l]
            
            # Get variable values
            vi = pyo_val(self.model.voltage_squared[i, 0])
            vj = pyo_val(self.model.voltage_squared[j, 0])
            p = pyo_val(self.model.active_power_flow[l, 0])
            q = pyo_val(self.model.reactive_power_flow[l, 0])
            i_sq = pyo_val(self.model.squared_current_magnitude[l, 0])
            
            # Calculate expected voltage drop
            lhs = vj
            rhs = vi - 2*(R*p + X*q) + (R**2 + X**2)*i_sq
            error = abs(lhs - rhs)
            
            if error > 1e-4:  # Tolerance for numerical issues
                voltage_drop_violations.append({
                    "line": l,
                    "from_bus": i,
                    "to_bus": j,
                    "v_i_squared": vi,
                    "v_j_squared": vj,
                    "p_flow": p,
                    "q_flow": q,
                    "i_squared": i_sq,
                    "lhs": lhs,
                    "rhs": rhs,
                    "error": error
                })
        
        # Log results
        if voltage_drop_violations:
            worst = max(voltage_drop_violations, key=lambda v: v["error"])
            self.logger.warning(f"Found {len(voltage_drop_violations)} voltage drop constraint violations")
            self.logger.warning(f"Worst violation on line {worst['line']} ({worst['from_bus']}->{worst['to_bus']}): error = {worst['error']:.6g}")
            
            # Log details of first few violations
            for violation in voltage_drop_violations[:5]:
                l = violation["line"]
                i, j = violation["from_bus"], violation["to_bus"]
                self.logger.warning(f"  Line {l} ({i}->{j}):")
                self.logger.warning(f"    vi²={violation['v_i_squared']:.6g}, vj²={violation['v_j_squared']:.6g}")
                self.logger.warning(f"    p={violation['p_flow']:.6g}, q={violation['q_flow']:.6g}, i²={violation['i_squared']:.6g}")
                self.logger.warning(f"    lhs={violation['lhs']:.6g}, rhs={violation['rhs']:.6g}, error={violation['error']:.6g}")
        else:
            self.logger.info("No voltage drop constraint violations found!")
            
        # Store violations for debugging
        self.debug_logs["voltage_drop_violations"] = voltage_drop_violations
    
    def verify_socp_cone_constraints(self):
        """
        Verify second-order cone constraints to ensure the model is working correctly.
        """
        self.logger.info("Verifying second-order cone constraints...")
        
        # Track violations
        socp_violations = []
        
        # Check each line
        for l in self.model.lines:
            # Only check active lines
            if pyo_val(self.model.line_status[l]) < 0.5:
                continue
                
            # Get bus index
            i = self.model.from_bus[l]
            
            # Get variable values for time 0
            p = pyo_val(self.model.active_power_flow[l, 0])
            q = pyo_val(self.model.reactive_power_flow[l, 0])
            vi = pyo_val(self.model.voltage_squared[i, 0])
            i_sq = pyo_val(self.model.squared_current_magnitude[l, 0])
            
            # Calculate norm squared
            norm_squared = (2*p)**2 + (2*q)**2 + (vi - i_sq)**2
            
            # Calculate right-hand side
            rhs_squared = (vi + i_sq)**2
            
            # Check if constraint is satisfied
            if norm_squared > rhs_squared * (1 + 1e-4):  # Allow small tolerance
                violation = {
                    "line": l,
                    "from_bus": i,
                    "p_flow": p,
                    "q_flow": q,
                    "v_i_squared": vi,
                    "i_squared": i_sq,
                    "norm_squared": norm_squared,
                    "rhs_squared": rhs_squared,
                    "violation": norm_squared - rhs_squared
                }
                socp_violations.append(violation)
        
        # Log results
        if socp_violations:
            worst = max(socp_violations, key=lambda v: v["violation"])
            self.logger.warning(f"Found {len(socp_violations)} SOC constraint violations")
            self.logger.warning(f"Worst violation on line {worst['line']}: violation = {worst['violation']:.6g}")
            
            # Log details for worst violations
            for violation in socp_violations[:5]:
                l = violation["line"]
                self.logger.warning(f"  Line {l} from bus {violation['from_bus']}:")
                self.logger.warning(f"    p={violation['p_flow']:.6g}, q={violation['q_flow']:.6g}")
                self.logger.warning(f"    vi²={violation['v_i_squared']:.6g}, i²={violation['i_squared']:.6g}")
                self.logger.warning(f"    ||.||²={violation['norm_squared']:.6g}, rhs²={violation['rhs_squared']:.6g}")
                self.logger.warning(f"    violation={violation['violation']:.6g}")
        else:
            self.logger.info("All SOC constraints satisfied within tolerance!")
            
        # Store violations for debugging
        self.debug_logs["socp_violations"] = socp_violations
    
    def verify_line_flow_bounds(self):
        """
        Verify line flow bound constraints to ensure the big-M constraints are working correctly.
        """
        self.logger.info("Verifying line flow bound constraints...")
        
        # Track violations
        flow_bound_violations = []
        
        # Check each line
        for l in self.model.lines:
            # Get line status
            status = pyo_val(self.model.line_status[l])
            
            # Get flow values
            p = pyo_val(self.model.active_power_flow[l, 0])
            q = pyo_val(self.model.reactive_power_flow[l, 0])
            
            # Get big-M
            big_M = pyo_val(self.model.big_M_flow[l])
            
            # Check active power upper/lower bounds
            if abs(p) > big_M * status + 1e-4:
                violation = {
                    "line": l,
                    "status": status,
                    "p_flow": p,
                    "q_flow": q,
                    "big_M": big_M,
                    "p_bound": big_M * status,
                    "violation": abs(p) - big_M * status,
                    "type": "active power"
                }
                flow_bound_violations.append(violation)
                
            # Check reactive power upper/lower bounds
            if abs(q) > big_M * status + 1e-4:
                violation = {
                    "line": l,
                    "status": status,
                    "p_flow": p,
                    "q_flow": q,
                    "big_M": big_M,
                    "q_bound": big_M * status,
                    "violation": abs(q) - big_M * status,
                    "type": "reactive power"
                }
                flow_bound_violations.append(violation)
        
        # Log results
        if flow_bound_violations:
            self.logger.warning(f"Found {len(flow_bound_violations)} flow bound violations")
            
            # Group by violation type
            p_violations = [v for v in flow_bound_violations if v["type"] == "active power"]
            q_violations = [v for v in flow_bound_violations if v["type"] == "reactive power"]
            
            if p_violations:
                worst_p = max(p_violations, key=lambda v: v["violation"])
                self.logger.warning(f"Worst active power bound violation on line {worst_p['line']}: {worst_p['violation']:.6g}")
                
            if q_violations:
                worst_q = max(q_violations, key=lambda v: v["violation"])
                self.logger.warning(f"Worst reactive power bound violation on line {worst_q['line']}: {worst_q['violation']:.6g}")
                
            # Log details for selected violations
            for violation in flow_bound_violations[:5]:
                l = violation["line"]
                self.logger.warning(f"  Line {l} ({violation['type']} violation):")
                if violation["type"] == "active power":
                    self.logger.warning(f"    |p|={abs(violation['p_flow']):.6g} > {violation['p_bound']:.6g} = M·status")
                else:
                    self.logger.warning(f"    |q|={abs(violation['q_flow']):.6g} > {violation['q_bound']:.6g} = M·status")
                self.logger.warning(f"    violation={violation['violation']:.6g}, status={violation['status']}")
        else:
            self.logger.info("All flow bound constraints satisfied within tolerance!")
            
        # Store violations for debugging
        self.debug_logs["flow_bound_violations"] = flow_bound_violations
    def verify_radiality_topology(self, optimizer, net_opt):
        """Verify if the resulting network topology is radial by checking graph properties"""
        import networkx as nx
        
        self.logger.info("Verifying radiality of resulting network topology...")
        
        # Create a graph from the optimized network
        G = nx.Graph()
        
        # Add all buses as nodes
        for b in net_opt.bus.index:
            G.add_node(b)
        
        # Add active lines as edges
        line_status = {}
        if hasattr(optimizer.model, 'line_status'):
            for l in optimizer.model.lines:
                status = round(pyo_val(optimizer.model.line_status[l]))
                line_status[l] = status
                
                # Add edge if line is active
                if status > 0.5:
                    from_bus = optimizer.model.from_bus[l]
                    to_bus = optimizer.model.to_bus[l]
                    G.add_edge(from_bus, to_bus)
        
        # Identify substations
        substations = list(optimizer.substations)
        
        # Check for cycles (a radial network shouldn't have any)
        has_cycles = False
        cycles = []
        try:
            # Get all cycles in the graph
            cycle_basis = list(nx.cycle_basis(G))
            if cycle_basis:
                has_cycles = True
                cycles = cycle_basis
                self.logger.warning(f"Found {len(cycles)} cycles in the network: {cycles}")
            else:
                self.logger.info("Network has no cycles!")
        except Exception as e:
            self.logger.warning(f"Error while checking for cycles: {e}")
        
        # Check connectivity (should be connected if we have one substation, 
        # or potentially multiple components if multiple substations)
        components = list(nx.connected_components(G))
        self.logger.info(f"Network has {len(components)} connected components")
        
        # Check if components match substations
        has_correct_components = False
        if len(components) == 1:
            self.logger.info("Network is fully connected - appropriate for single substation")
            has_correct_components = True
        elif len(components) <= len(substations):
            self.logger.info(f"Network has {len(components)} components - appropriate for {len(substations)} substations")
            has_correct_components = True
        else:
            self.logger.warning(f"Network has {len(components)} components, more than the {len(substations)} substations")
        
        # Check for tree structure - a radial network should have n nodes and n-1 edges
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()
        expected_edges = total_nodes - len(components)  # One less than nodes for each component
        
        self.logger.info(f"Graph has {total_nodes} nodes and {total_edges} edges. Expected {expected_edges} edges for a forest.")
        
        is_tree_structure = total_edges == expected_edges
        if is_tree_structure:
            self.logger.info("Network forms a proper tree/forest structure!")
        else:
            self.logger.warning(f"Network does not form a tree/forest structure. Expected {expected_edges} edges, got {total_edges}.")
        
        # Final radiality assessment
        is_radial = (not has_cycles) and is_tree_structure and has_correct_components
        
        self.logger.info(f"Radiality assessment: {'Radial' if is_radial else 'Not Radial'}")
        
        return {
            "is_radial": is_radial,
            "has_cycles": has_cycles, 
            "is_tree_structure": is_tree_structure,
            "has_correct_components": has_correct_components,
            "num_components": len(components),
            "num_nodes": total_nodes,
            "num_edges": total_edges,
            "expected_edges": expected_edges
        }

    def extract_scf_flows(self):
        """Extract SCF flow values for debugging"""
        scf_flows = {}
        
        if hasattr(self.model, 'scf_flow_var'):
            for (i, j, l) in self.model.arcs:
                flow_value = pyo_val(self.model.scf_flow_var[i, j, l])
                if abs(flow_value) > 1e-6:  # Only store non-zero flows
                    scf_flows[(i, j, l)] = flow_value
        
        return scf_flows

    def verify_scf_radiality(self):
        """Verify SCF solution forms a tree"""
        model = self.model
        logger = self.logger
        
        if not hasattr(model, 'scf_flow_var'):
            logger.error("No SCF flow variables found in model!")
            return False
        
        # Extract flows
        scf_flows = self.extract_scf_flows()
        
        # Build directed graph from flows
        import networkx as nx
        G = nx.DiGraph()
        
        # Add all buses
        for b in model.buses:
            G.add_node(b)
        
        # Add edges where flow exists
        lines_with_flow = set()
        for (i, j, l), flow in scf_flows.items():
            if flow > 1e-6:
                G.add_edge(i, j, line=l, flow=flow)
                lines_with_flow.add(l)
        
        # Check 1: Number of edges should equal n-1
        n_nodes = len([b for b in model.buses if pyo_val(model.is_up[b]) > 0.5])
        n_edges = G.number_of_edges()
        
        logger.info(f"SCF Debug: {n_nodes} energized nodes, {n_edges} edges with flow")
        
        if n_edges != n_nodes - 1:
            logger.error(f"SCF Error: Expected {n_nodes-1} edges, got {n_edges}")
            return False
        
        # Check 2: Graph should be connected
        if not nx.is_weakly_connected(G):
            logger.error("SCF Error: Graph is not connected")
            return False
        
        # Check 3: No cycles
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                logger.error(f"SCF Error: Found cycles: {cycles}")
                return False
        except:
            pass
        
        # Check 4: Each line should have flow in only one direction
        line_directions = {}
        for (i, j, l), flow in scf_flows.items():
            if flow > 1e-6:
                if l in line_directions:
                    logger.error(f"SCF Error: Line {l} has flow in multiple directions!")
                    logger.error(f"  Flow {line_directions[l]} and flow ({i},{j})")
                    return False
                line_directions[l] = (i, j)
        
        # Check 5: Active lines match lines with flow
        active_lines = set()
        for l in model.lines:
            if pyo_val(model.line_status[l]) > 0.5:
                active_lines.add(l)
        
        if active_lines != lines_with_flow:
            logger.error(f"SCF Error: Active lines {active_lines} don't match lines with flow {lines_with_flow}")
            return False
        
        logger.info("SCF radiality verification passed!")
        return True


    def process_solution(self, update_network=True, tolerance=1e-5, output_dir=None):
        # after solve
        #for b in  self.model.buses:
        #    print(f"Bus {b}: inj={self.bus_p_inj[b]:.4f}, out-in={pyo_val(self.sum_outflow[b]-self.sum_inflow[b]):.4f}, slack={pyo_val(self.active_power_slack[b,0]):.4f}")
        # Set default output directory
        if output_dir is None:
            output_dir = Path("data_generation") / "logs" / "lp_files"
        else:
            output_dir = Path(output_dir)
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        lg = self.logger
        result = {
            'feasible': True,
            'obj_value': None,
            'solve_time': self.solve_time,
            'violations': [],
            'switches_changed': 0,
            'network_properties': {}
        }
        
        # Initialize list to collect all infeasibility messages
        infeasibility_messages = []
        
        # Check if model and solver results exist
        if self.model is None:
            msg = "No model exists. Call create_model() and solve() first."
            lg.error(msg)
            infeasibility_messages.append(msg)
            result['feasible'] = False
            return result
        
        if self.solver_results is None:
            msg = "No solver results. Call solve() first."
            lg.error(msg)
            infeasibility_messages.append(msg)
            result['feasible'] = False
            return result
        
        # Check termination status
        from pyomo.opt import check_optimal_termination
        is_optimal = check_optimal_termination(self.solver_results)
        if not is_optimal:
            termination_status = str(self.solver_results.solver.termination_condition)
            msg = f"Solver terminated with non-optimal status: {termination_status}"
            lg.warning(msg)
            infeasibility_messages.append(msg)
            result['feasible'] = False
            result['violations'].append({
                "type": "termination",
                "cond": termination_status
            })
        
        # Get objective value
        try:
            result['obj_value'] = pyo_val(self.model.objective)
            
            # Print objective breakdown for debugging
            lg.info("Objective breakdown:")
            lg.info(f"  Loss Term Value : {pyo_val(self.model.loss_term_expr):.6f}")
            lg.info(f"  voltagebounds slack Term Value: {pyo_val(self.model.voltage_bounds_slack_term_rule):.6f}")
            lg.info(f"  Load shed cost: {pyo_val(self.model.load_shed_cost):.6f}")
            lg.info(f"  voltage dropslack penalty:{pyo_val(self.model.voltage_drop_slack_term_rule)}")
            if self.toggles.get("include_switch_penalty", False):
                lg.info(f"  Switch Penalty Term Value: {pyo_val(self.model.switch_pen_expr):.6f}")
            lg.info(f"  Total Objective Value: {result['obj_value']:.6f}")
        except Exception as e:
            msg = f"Error retrieving objective value: {e}"
            lg.error(msg)
            infeasibility_messages.append(msg)
            result['feasible'] = False
        
        # Check constraint violations
        constraint_violations = []
        for c in self.model.component_data_objects(Constraint, active=True):
            try:
                lb = c.lower if c.has_lb() else None
                ub = c.upper if c.has_ub() else None
                lhs = pyo_val(c.body)
                if (lb is not None and lhs < lb - tolerance) or (ub is not None and lhs > ub + tolerance):
                    violation = {
                        "type": "constraint",
                        "name": c.name,
                        "value": lhs,
                        "lb": lb,
                        "ub": ub
                    }
                    result['violations'].append(violation)
                    
                    if lb is not None and lhs < lb - tolerance:
                        violation_msg = f"Constraint violation: {c.name} = {lhs:.6g} < {lb} (lower bound) by {lb - lhs:.6g}"
                    else:
                        violation_msg = f"Constraint violation: {c.name} = {lhs:.6g} > {ub} (upper bound) by {lhs - ub:.6g}"
                    
                    constraint_violations.append(violation_msg)
                    lg.debug(violation_msg)
            except Exception as e:
                msg = f"Error checking constraint {c.name}: {e}"
                lg.warning(msg)
                constraint_violations.append(msg)
        
        if constraint_violations:
            infeasibility_messages.append("\n=== CONSTRAINT VIOLATIONS ===")
            infeasibility_messages.extend(constraint_violations)
        
        # Check variable domain violations (binary variables, bounds)
        binary_violations = []
        for v in self.model.component_data_objects(Var, active=True):
            try:
                val = pyo_val(v)
                if v.domain is Binary and abs(val - round(val)) > tolerance:
                    violation = {
                        "type": "domain",
                        "var": v.name,
                        "value": val
                    }
                    result['violations'].append(violation)
                    
                    violation_msg = f"Binary variable violation: {v.name} = {val:.6g} (not binary)"
                    binary_violations.append(violation_msg)
                    lg.debug(violation_msg)
            except Exception as e:
                msg = f"Error checking variable {v.name}: {e}"
                lg.warning(msg)
                binary_violations.append(msg)
        
        if binary_violations:
            infeasibility_messages.append("\n=== BINARY VARIABLE VIOLATIONS ===")
            infeasibility_messages.extend(binary_violations)
        
        # Update network if requested
        if update_network:
            updated_network = self.net.deepcopy()
            switches_changed = self._update_switch_status(updated_network)
            result['switches_changed'] = switches_changed
            
            # Apply voltage results directly
            if hasattr(self.model, 'voltage_squared'):
                # Create res_bus if it doesn't exist
                if not hasattr(updated_network, 'res_bus') or updated_network.res_bus.empty:
                    updated_network.res_bus = pd.DataFrame(index=updated_network.bus.index)
                    updated_network.res_bus['vm_pu'] = np.nan
                    updated_network.res_bus['va_degree'] = 0.0
                
                # Update voltage magnitudes
                for b in self.model.buses:
                    try:
                        v_squared = pyo_val(self.model.voltage_squared[b, 0])
                        updated_network.res_bus.at[b, 'vm_pu'] = np.sqrt(max(0, v_squared))  # Ensure positive
                    except Exception as e:
                        msg = f"Could not apply voltage result for bus {b}: {e}"
                        lg.warning(msg)
                        infeasibility_messages.append(msg)
            
            # Check network topology
            rad, con = is_radial_and_connected(updated_network, y_mask=self.active_bus)
            result['network_properties']['radial'] = rad
            result['network_properties']['connected'] = con
            
            if not rad or not con:
                result['feasible'] = False
                result['violations'].append({
                    "type": "topology",
                    "radial": rad,
                    "connected": con
                })
                
                topology_msg = f"Network topology issues: radial={rad}, connected={con}"
                lg.warning(topology_msg)
                infeasibility_messages.append("\n=== TOPOLOGY VIOLATIONS ===")
                infeasibility_messages.append(topology_msg)
                
                # Additional topology analysis to identify specific issues
                import networkx as nx
                
                G = nx.Graph()
                for i, line in updated_network.line[updated_network.line.in_service].iterrows():
                    G.add_edge(line.from_bus, line.to_bus)
                
                # Add edges for closed switches
                for i, sw in updated_network.switch[(updated_network.switch.et == 'l') & (updated_network.switch.closed)].iterrows():
                    line = updated_network.line.loc[sw.element]
                    G.add_edge(line.from_bus, line.to_bus)
                
                # Check components
                components = list(nx.connected_components(G))
                num_components = len(components)
                infeasibility_messages.append(f"Number of connected components: {num_components}")
                
                if num_components > 1:
                    comp_sizes = sorted([len(c) for c in components], reverse=True)
                    infeasibility_messages.append(f"Component sizes: {comp_sizes}")
                
                # Check for cycles
                cycles = []
                for component in components:
                    subgraph = G.subgraph(component)
                    try:
                        cycle = nx.find_cycle(subgraph)
                        if cycle:
                            cycles.append(cycle)
                            infeasibility_messages.append(f"Cycle found in component: {cycle}")
                    except nx.NetworkXNoCycle:
                        pass
                
                if cycles:
                    infeasibility_messages.append(f"Total cycles found: {len(cycles)}")
            
            # Apply line flows and check for overloads
            if hasattr(self.model, 'active_power_flow') and hasattr(self.model, 'squared_current_magnitude'):
                # Create results tables if they don't exist
                if not hasattr(updated_network, 'res_line') or updated_network.res_line.empty:
                    updated_network.res_line = pd.DataFrame(index=updated_network.line.index)
                    for col in ['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar', 'pl_mw', 'i_ka']:
                        updated_network.res_line[col] = 0.0
                
                # Update line flows and check for overloads
                thermal_violations = []
                loss_mw = 0
                for l in self.model.lines:
                    try:
                        # Active and reactive power flows
                        p_val = pyo_val(self.model.active_power_flow[l, 0]) * self.S_base_VA / 1e6  # Convert to MW
                        q_val = pyo_val(self.model.reactive_power_flow[l, 0]) * self.S_base_VA / 1e6  # Convert to MVar
                        i_squared = pyo_val(self.model.squared_current_magnitude[l, 0])
                        r_pu = pyo_val(self.model.line_resistance_pu[l])
                        
                        # Calculate loss
                        loss = r_pu * i_squared * self.S_base_VA / 1e6  # Convert to MW
                        
                        # Apply to network results
                        updated_network.res_line.at[l, 'p_from_mw'] = p_val
                        updated_network.res_line.at[l, 'q_from_mvar'] = q_val
                        updated_network.res_line.at[l, 'p_to_mw'] = -p_val + loss
                        updated_network.res_line.at[l, 'q_to_mvar'] = -q_val
                        updated_network.res_line.at[l, 'pl_mw'] = loss
                        
                        # Calculate current in kA if possible
                        i_ka = None
                        if 'max_i_ka' in updated_network.line.columns and l in updated_network.line.index:
                            max_i_ka = updated_network.line.at[l, 'max_i_ka']
                            vn_kv = updated_network.line.at[l, 'vn_kv'] if 'vn_kv' in updated_network.line.columns else None
                            if vn_kv is not None and vn_kv > 0:
                                s_mva = np.sqrt(p_val**2 + q_val**2)
                                i_ka = s_mva / (np.sqrt(3) * vn_kv)
                                updated_network.res_line.at[l, 'i_ka'] = i_ka
                                
                                # Check for thermal overload
                                if max_i_ka > 0 and i_ka > max_i_ka * (1 + tolerance):
                                    loading_pct = 100 * i_ka / max_i_ka
                                    violation_msg = f"Thermal overload: Line {l} at {loading_pct:.1f}% of rating (I={i_ka:.4f} kA, Max={max_i_ka:.4f} kA)"
                                    thermal_violations.append(violation_msg)
                                    result['violations'].append({
                                        "type": "thermal",
                                        "line": int(l),
                                        "loading": float(i_ka / max_i_ka)
                                    })
                        
                        loss_mw += loss
                    except Exception as e:
                        msg = f"Could not apply line flow results for line {l}: {e}"
                        lg.warning(msg)
                        infeasibility_messages.append(msg)
                
                if thermal_violations:
                    infeasibility_messages.append("\n=== THERMAL OVERLOAD VIOLATIONS ===")
                    infeasibility_messages.extend(thermal_violations)
                
                result['network_properties']['loss_mw'] = loss_mw
            
            # Check voltage limits
            voltage_violations = []
            vmin, vmax = 0.90, 1.10  # Standard voltage limits
            
            if hasattr(updated_network, 'res_bus') and 'vm_pu' in updated_network.res_bus.columns:
                vm = updated_network.res_bus['vm_pu']
                v_min_actual = vm.min() if not vm.empty else None
                v_max_actual = vm.max() if not vm.empty else None
                
                if v_min_actual is not None and v_max_actual is not None:
                    result['network_properties']['voltage_range'] = (v_min_actual, v_max_actual)
                    
                    # Check under-voltage buses
                    under_voltage_buses = updated_network.res_bus[vm < vmin - tolerance]
                    if not under_voltage_buses.empty:
                        for b, v in under_voltage_buses['vm_pu'].items():
                            violation_msg = f"Under-voltage: Bus {b} at {v:.4f} p.u. (min={vmin})"
                            voltage_violations.append(violation_msg)
                    
                    # Check over-voltage buses
                    over_voltage_buses = updated_network.res_bus[vm > vmax + tolerance]
                    if not over_voltage_buses.empty:
                        for b, v in over_voltage_buses['vm_pu'].items():
                            violation_msg = f"Over-voltage: Bus {b} at {v:.4f} p.u. (max={vmax})"
                            voltage_violations.append(violation_msg)
                    
                    if under_voltage_buses.shape[0] > 0 or over_voltage_buses.shape[0] > 0:
                        result['feasible'] = False
                        result['violations'].append({
                            "type": "voltage",
                            "min": v_min_actual,
                            "max": v_max_actual,
                            "band": (vmin, vmax),
                            "under_voltage_count": under_voltage_buses.shape[0],
                            "over_voltage_count": over_voltage_buses.shape[0]
                        })
                        infeasibility_messages.append("\n=== VOLTAGE VIOLATIONS ===")
                        infeasibility_messages.extend(voltage_violations)
            
            # Store the updated network
            self.net_result = updated_network
        self.logger.info("Power flow balance at each bus:")
        for b in self.model.buses:
            try:
                inflow = pyo_val(self.model.sum_inflow[b])
                outflow = pyo_val(self.model.sum_outflow[b])
                p_inj = self.bus_p_inj[b]
                flow_value = outflow - inflow
                
                if b in self.substations:
                    expected = len(self.model.partitionedbuses) * pyo_val(self.model.is_up[b])
                    self.logger.info(f"Bus {b} (SUBSTATION): inj=N/A, out-in={flow_value:.4f}, expected={expected:.4f}")
                else:
                    expected = p_inj * pyo_val(self.model.is_up[b])
                    slack = pyo_val(self.model.active_power_slack[b,0]) if hasattr(self.model, 'active_power_slack') else 0
                    self.logger.info(f"Bus {b}: inj={p_inj:.4f}, out-in={flow_value:.4f}, expected={expected:.4f}, slack={slack:.4f}")
            except Exception as e:
                self.logger.error(f"Error calculating flow for bus {b}: {e}")
        # Check radiality flows specifically
    
        if self.toggles.get("include_radiality_constraints", False):
            if self.toggles.get("use_root_flow", False) and hasattr(self.model, 'branch_flow'):
                self.logger.info("Checking radiality flow constraints for single commodity flow model...")
                
                # Track which buses are reachable from substations
                reachable_buses = set()
                
                # Check flow from substations
                for b in self.substations:
                    if b in self.model.buses:
                        # For the single commodity flow model
                        outgoing_lines = [l for l in self.model.lines if self.model.from_bus[l] == b and pyo_val(self.model.line_status[l]) > 0.5]
                        incoming_lines = [l for l in self.model.lines if self.model.to_bus[l] == b and pyo_val(self.model.line_status[l]) > 0.5]
                        
                        outflow = sum(pyo_val(self.model.branch_flow[l]) for l in outgoing_lines if hasattr(self.model, 'branch_flow'))
                        inflow = sum(pyo_val(self.model.branch_flow[l]) for l in incoming_lines if hasattr(self.model, 'branch_flow'))
                        
                        net_outflow = outflow - inflow
                        self.logger.info(f"Substation {b} net flow outbound: {net_outflow:.4f}")
            
            elif self.toggles.get("use_parent_child_radiality", False) and hasattr(self.model, 'arc_in_tree'):
                self.logger.info("Checking parent-child radiality constraints...")
                
                # Check if arcs are defined
                if hasattr(self.model, 'arcs'):
                    # Count active arcs
                    active_arcs = sum(1 for (i,j,l) in self.model.arcs 
                                    if pyo_val(self.model.arc_in_tree[i,j,l]) > 0.5)
                    
                    # Count active buses
                    active_buses = sum(1 for b in self.model.buses 
                                    if b in self.model.partitionedbuses and pyo_val(self.model.is_up[b]) > 0.5)
                    
                    self.logger.info(f"Parent-child radiality check: {active_arcs} active arcs, {active_buses} active buses")
                else:
                    self.logger.warning("Parent-child formulation is enabled but 'arcs' not found in model")
            
            else:
                self.logger.info("Basic radiality checking: using total line and bus counts")
            
            # Always do basic radiality check by counting lines and buses
            active_lines = sum(1 for l in self.model.lines if pyo_val(self.model.line_status[l]) > 0.5)
            energized_buses = sum(1 for b in self.model.buses 
                                if b in self.model.partitionedbuses and pyo_val(self.model.is_up[b]) > 0.5)
            active_substations = sum(1 for b in self.substations if b in self.model.buses)
            
            self.logger.info(f"Basic radiality check: {active_lines} active lines, {energized_buses + active_substations} energized buses")
            expected_lines = energized_buses + active_substations - 1
            
            if active_lines == expected_lines:
                self.logger.info(f"Network forms a proper tree structure (n-1 lines)")
            else:
                self.logger.warning(f"Network structure issue: expected {expected_lines} lines for a tree, found {active_lines}")
        # Always do basic radiality check by counting lines and buses
        active_lines = sum(1 for l in self.model.lines if pyo_val(self.model.line_status[l]) > 0.5)
        energized_buses = sum(1 for b in self.model.buses if b in self.model.partitionedbuses 
                            and pyo_val(self.model.is_up[b]) > 0.5)
        active_substations = sum(1 for b in self.substations if b in self.model.buses)
        
        self.logger.info(f"Basic radiality check: {active_lines} active lines, {energized_buses + active_substations} energized buses")
        expected_lines = energized_buses + active_substations - 1
        
        if active_lines == expected_lines:
            self.logger.info(f"Network forms a proper tree structure (n-1 lines)")
        else:
            self.logger.warning(f"Network structure issue: expected {expected_lines} lines for a tree, found {active_lines}")
        # Continue with the rest of the method...
        # Overall assessment
        if result['violations']:
            lg.warning(f"Found {len(result['violations'])} violations in solution")
            result['feasible'] = False
        else:
            lg.info("Solution is feasible with no violations detected")
        
        # Write infeasibility log file if there are any issues
        if infeasibility_messages:
            graph_id = self.id or "unknown"
            infeas_file = output_dir / f"{graph_id}_infeasibilities.txt"
            
            with open(infeas_file, 'w') as f:
                f.write(f"{graph_id}: Infeasibilities\n")
                f.write("="*50 + "\n")
                f.write(f"Solution time: {self.solve_time:.2f} seconds\n")
                f.write(f"Objective value: {result.get('obj_value', 'N/A')}\n")
                f.write("="*50 + "\n\n")
                f.write("\n".join(infeasibility_messages))
            
            lg.info(f"Infeasibilities log written to: {infeas_file}")
        
        return result


def get_reactive_injection(df, bus, assumed_pf=0.9):
    sub = df.loc[df.bus == bus]
    if sub.empty:
        return 0

    if "q_mvar" in df.columns:
        non_null = sub["q_mvar"].count()
        total = len(sub)
        if non_null >= total * 0.5:
            derived = sub["p_mw"] * np.tan(np.arccos(assumed_pf))
            q_values = sub["q_mvar"].fillna(derived)
            return q_values.sum()
        else:
            return (sub["p_mw"] * np.tan(np.arccos(assumed_pf))).sum()
    else:
        return (sub["p_mw"] * np.tan(np.arccos(assumed_pf))).sum()