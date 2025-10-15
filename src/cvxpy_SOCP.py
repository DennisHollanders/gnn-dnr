import cvxpy as cp
import numpy as np
import pandas as pd 
import logging
import math

from sympy import N

def build_misocp_problem(net, toggles, logger):
    if logger is None:
        logger =logging.getLogger(__name__)

     # Default toggles
    if toggles is None:
        toggles = {"include_voltage_drop_constraint": True, 
           "include_voltage_bounds_constraint": True, 
           "include_power_balance_constraint": True, 
           "include_radiality_constraints": True, 
           "use_spanning_tree_radiality": False, 
           "use_root_flow": True, 
           "use_parent_child_radiality": False, 
           "include_switch_penalty": True, 
           "allow_load_shed": True, 
           "all_lines_are_switches": True, "include_cone_constraint": True}
        
    logger.info(f"MISOCP toggles: {toggles}")

   # Initialize network data - following SOCP_class_dnr naming
    S_base_MVA = 1.0
    S_base_VA = S_base_MVA * 1e6
    
    # calculate per-unit impedances to ensure they exist.
    line_with_pu = net.line.copy()
    bus_vn_kv = net.bus.vn_kv
    v_base_ln = (bus_vn_kv * 1e3) / math.sqrt(3)
    z_base = (v_base_ln**2) / S_base_VA
    line_z_base = line_with_pu['from_bus'].map(z_base)
    line_with_pu['r_pu'] = (line_with_pu['r_ohm_per_km'] * line_with_pu['length_km']) / line_z_base
    line_with_pu['x_pu'] = (line_with_pu['x_ohm_per_km'] * line_with_pu['length_km']) / line_z_base
    
    # Map buses and lines t
    bus_list = list(net.bus.index)
    bus_map = {b: i for i, b in enumerate(bus_list)}
    line_list = list(net.line.index)
    
    N = len(bus_list)  
    E = len(line_list)  
        
    # Extract line parameters 
    from_bus_indices = np.array([bus_map[net.line.at[l, 'from_bus']] for l in line_list], dtype=int)
    to_bus_indices = np.array([bus_map[net.line.at[l, 'to_bus']] for l in line_list], dtype=int)
    
    # Get the pre-calculated per-unit values
    r_pu = np.array(line_with_pu.loc[line_list, 'r_pu'])
    x_pu = np.array(line_with_pu.loc[line_list, 'x_pu'])

    p_inj = np.zeros(N)
    q_inj = np.zeros(N)
    S_base_MVA = 1.0

    for b_pp_idx in bus_list:
        i_cvx_idx = bus_map[b_pp_idx]
        p_mw_val = (net.gen[net.gen.bus == b_pp_idx].p_mw.sum() - 
                    net.load[net.load.bus == b_pp_idx].p_mw.sum())
        q_mvar_val = (get_reactive_injection(net.gen, b_pp_idx) -
                      get_reactive_injection(net.load, b_pp_idx))
        p_inj[i_cvx_idx] = p_mw_val / S_base_MVA
        q_inj[i_cvx_idx] = q_mvar_val / S_base_MVA


    # Big-M values for power flows
    bigM_flow = np.zeros(E)
    for idx, l_pp_idx in enumerate(line_list):
        row = net.line.loc[l_pp_idx]
        if 'max_i_ka' in row and pd.notna(row.max_i_ka) and row.max_i_ka > 0:
            if row.from_bus in net.bus.index:
                v_ln_kv = net.bus.vn_kv[row.from_bus]
                s_max_mva_thermal = np.sqrt(3) * v_ln_kv * row.max_i_ka
                bigM_flow[idx] = s_max_mva_thermal / S_base_MVA
            else:
                bigM_flow[idx] = np.sum(np.abs(p_inj)) if np.sum(np.abs(p_inj)) > 0 else 10.0
        else:
            bigM_flow[idx] = np.sum(np.abs(p_inj)) if np.sum(np.abs(p_inj)) > 0 else 10.0
    
    # Voltage bounds
    v_upper, v_lower = 1.10, 0.90
    bigM_v = (v_upper**2 - v_lower**2) * 1.5

    # Identify substations
    substation_cvx_indices = [bus_map[b_pp_idx] for b_pp_idx in net.ext_grid.bus if b_pp_idx in bus_map]
    
    # Get initial line status for switch penalty
    initial_line_status = np.ones(E)
    for idx, l_pp_idx in enumerate(line_list):
        line_switches = net.switch[(net.switch.et == 'l') & (net.switch.element == l_pp_idx)]
        if not line_switches.empty:
            initial_line_status[idx] = 1 if line_switches.closed.all() else 0

    # Build directed arcs for SCF
    arcs = [(u,v,e) for e,(u,v) in enumerate(zip(from_bus_indices,to_bus_indices))]
    arcs += [(v,u,e) for e,(u,v) in enumerate(zip(from_bus_indices,to_bus_indices))]
    arc_u, arc_v, arc_e = map(np.array, zip(*arcs))
    A = len(arcs)
    

    # continious Variables
    v_sq = cp.Variable(N, name='v_sq', nonneg=True)  # squared voltages
    p_flow = cp.Variable(E, name='p_flow')  # active flows
    q_flow = cp.Variable(E, name='q_flow')  # reactive flows
    I_sq = cp.Variable(E, name='I_sq', nonneg=True)  # squared current magnitudes
    
    
    # Binary variables
    y_line = cp.Variable(E, name='y_line', boolean=True)  # y_ij: line status
    z_bus = cp.Variable(N, name='z_bus', boolean=True)  # z_i: bus energization
  
    # SCF variables for radiality
    if toggles.get("include_radiality_constraints") and toggles.get("use_root_flow"):
        # Build directed arcs for SCF
        arcs = [(u, v, e) for e, (u, v) in enumerate(zip(from_bus_indices, to_bus_indices))]
        arcs += [(v, u, e) for e, (u, v) in enumerate(zip(from_bus_indices, to_bus_indices))]
        A = len(arcs)
        
        arc_u = np.array([u for u, v, e in arcs], dtype=int)
        arc_v = np.array([v for u, v, e in arcs], dtype=int)
        arc_e = np.array([e for u, v, e in arcs], dtype=int)
        
        # f_a: commodity flow variables
        f_flow = cp.Variable(A, name='f_flow', nonneg=True)

    
    # ----------------------------------------------------------------------------
    # Create constraints
    # ----------------------------------------------------------------------------

    # Constraints list
    constraints = []
    # 1. Substations always energized with fixed voltage
    for i_cvx_idx in substation_cvx_indices:
        constraints.append(z_bus[i_cvx_idx] == 1)
        sub_pp_idx = bus_list[i_cvx_idx]
        sub_vm_pu = net.ext_grid[net.ext_grid.bus == sub_pp_idx].vm_pu.iloc[0]
        constraints.append(v_sq[i_cvx_idx] == sub_vm_pu**2)
    
    # 2. Load shedding constraints
    if not toggles.get("allow_load_shed", False):
        for i_cvx_idx in range(N):
            if i_cvx_idx not in substation_cvx_indices:
                constraints.append(z_bus[i_cvx_idx] == 1)
    
    # 3. Voltage bounds
    if toggles.get("include_voltage_bounds_constraint", True):
        for i_cvx_idx in range(N):
            if i_cvx_idx not in substation_cvx_indices:
                # Upper bound: V_i ≤ V_max² + M_v(1-z_i)
                constraints.append(
                    v_sq[i_cvx_idx] <= v_upper**2 + bigM_v * (1 - z_bus[i_cvx_idx])
                )
                # Lower bound: V_i ≥ V_min² * z_i
                constraints.append(
                    v_sq[i_cvx_idx] >= v_lower**2 * z_bus[i_cvx_idx]
                )
    
    # 4. Power flow bounds (Big-M formulation)
    for e_cvx_idx in range(E):
        M = bigM_flow[e_cvx_idx]
        constraints.extend([
            p_flow[e_cvx_idx] <= M * y_line[e_cvx_idx],
            p_flow[e_cvx_idx] >= -M * y_line[e_cvx_idx],
            q_flow[e_cvx_idx] <= M * y_line[e_cvx_idx],
            q_flow[e_cvx_idx] >= -M * y_line[e_cvx_idx],
            I_sq[e_cvx_idx] <= M**2 * y_line[e_cvx_idx]
        ])
    
    # 5. SOCP cone constraints: P² + Q² ≤ V_i * I
    if toggles.get("include_cone_constraint", True):
        for e_cvx_idx in range(E):
            u_idx = from_bus_indices[e_cvx_idx]
            M = bigM_flow[e_cvx_idx]
            constraints.append(
                cp.norm(cp.vstack([
                    2 * p_flow[e_cvx_idx], 
                    2 * q_flow[e_cvx_idx], 
                    v_sq[u_idx] - I_sq[e_cvx_idx]
                ]), 2) <= v_sq[u_idx] + I_sq[e_cvx_idx] + 2*M**2 * (1 - y_line[e_cvx_idx])
            )
    
    # 6. Voltage drop constraints
    if toggles.get("include_voltage_drop_constraint", True):
        for e_cvx_idx in range(E):
            u_idx = from_bus_indices[e_cvx_idx]
            v_idx = to_bus_indices[e_cvx_idx]
            R_l = r_pu[e_cvx_idx]
            X_l = x_pu[e_cvx_idx]
            
            # V_j = V_i - 2(R*P + X*Q) + (R²+X²)*I
            voltage_drop = (v_sq[v_idx] - v_sq[u_idx] + 
                          2 * (R_l * p_flow[e_cvx_idx] + X_l * q_flow[e_cvx_idx]) - 
                          (R_l**2 + X_l**2) * I_sq[e_cvx_idx])
            
            constraints.extend([
                voltage_drop <= bigM_v * (1 - y_line[e_cvx_idx]),
                voltage_drop >= -bigM_v * (1 - y_line[e_cvx_idx])
            ])
    
    # 7. Line-bus linking constraints
    for e_cvx_idx in range(E):
        constraints.extend([
            y_line[e_cvx_idx] <= z_bus[from_bus_indices[e_cvx_idx]],
            y_line[e_cvx_idx] <= z_bus[to_bus_indices[e_cvx_idx]]
        ])
    
    # 8. Bus connectivity constraints
    if toggles.get("include_radiality_constraints", True):
        for i_cvx_idx in range(N):
            if i_cvx_idx not in substation_cvx_indices:
                incident_lines = [e for e in range(E) if 
                                from_bus_indices[e] == i_cvx_idx or 
                                to_bus_indices[e] == i_cvx_idx]
                if incident_lines:
                    constraints.append(
                        cp.sum([y_line[e] for e in incident_lines]) >= z_bus[i_cvx_idx]
                    )
    
    # 9. Power balance constraints
    if toggles.get("include_power_balance_constraint", True):
        for i_cvx_idx in range(N):
            if i_cvx_idx not in substation_cvx_indices:
                # Active power balance
                p_out_lines = [e for e in range(E) if from_bus_indices[e] == i_cvx_idx]
                p_in_lines = [e for e in range(E) if to_bus_indices[e] == i_cvx_idx]
                
                p_out = cp.sum([p_flow[e] for e in p_out_lines]) if p_out_lines else 0
                p_in = cp.sum([p_flow[e] for e in p_in_lines]) if p_in_lines else 0
                
                constraints.append(p_out - p_in == p_inj[i_cvx_idx] * z_bus[i_cvx_idx])
                
                # Reactive power balance
                q_out = cp.sum([q_flow[e] for e in p_out_lines]) if p_out_lines else 0
                q_in = cp.sum([q_flow[e] for e in p_in_lines]) if p_in_lines else 0
                
                constraints.append(q_out - q_in == q_inj[i_cvx_idx] * z_bus[i_cvx_idx])

    # 10. SCF radiality constraints
    if toggles.get("include_radiality_constraints", True) and toggles.get("use_root_flow", True):
        print("Using SCF radiality constraints")
        # SCF capacity constraints
        M_scf = N - 1

        for a_idx in range(A):
            constraints.append(f_flow[a_idx] <= M_scf * y_line[arc_e[a_idx]])
        
        # SCF conservation
        root_idx = substation_cvx_indices[0] if substation_cvx_indices else 0
        
        for i_cvx_idx in range(N):
            in_arcs = [a for a in range(A) if arc_v[a] == i_cvx_idx]
            out_arcs = [a for a in range(A) if arc_u[a] == i_cvx_idx]
            
            f_in = cp.sum([f_flow[a] for a in in_arcs]) if in_arcs else 0
            f_out = cp.sum([f_flow[a] for a in out_arcs]) if out_arcs else 0
            
            if i_cvx_idx == root_idx:
                constraints.append(f_out - f_in == cp.sum(z_bus) - 1)
            else:
                constraints.append(f_in - f_out == z_bus[i_cvx_idx])
        flow_dir = cp.Variable(E, boolean=True, name='flow_dir')  

        for a_idx, (u, v, e_idx) in enumerate(arcs):
            if a_idx < E:   
                constraints.append(f_flow[a_idx] <= M_scf * flow_dir[e_idx])
            else:
                constraints.append(f_flow[a_idx] <= M_scf * (1 - flow_dir[e_idx]))
        

        # Spanning tree constraint
        constraints.append(cp.sum(y_line) == cp.sum(z_bus) - 1)
    
    # ==================================
    # Objective Function
    # ==================================

    # Objective function
    loss_objective = cp.sum(cp.multiply(r_pu, I_sq))
    
    # Add switch penalty if enabled
    if toggles.get("include_switch_penalty", True):
        switch_penalty = 0.0001  
        switch_changes = cp.sum(cp.abs(y_line  - initial_line_status))
        loss_objective += switch_penalty * switch_changes
    
    # Add load shedding penalty if enabled
    if toggles.get("allow_load_shed", False):
        load_shed_penalty = 100  
        load_buses = [i for i in range(N) if i not in substation_cvx_indices and p_inj[i] < 0]
        shed_penalty = cp.sum([cp.abs(p_inj[i]) * (1 - z_bus[i]) for i in load_buses])
        loss_objective += load_shed_penalty * shed_penalty
    
    # Create the optimization problem
    problem = cp.Problem(cp.Minimize(loss_objective), constraints)

    # Store variables
    variables = {
        'v_sq': v_sq,
        'p_flow': p_flow,
        'q_flow': q_flow,
        'I_sq': I_sq,
        'y_line': y_line,
        'z_bus': z_bus,
    }
    print(f"Variables: {variables.keys()}")
    if toggles.get("include_radiality_constraints") and toggles.get("use_root_flow"):
        variables['f_flow'] = f_flow
        variables['flow_dir'] = flow_dir
        variables['arc_u'] = arc_u
        variables['arc_v'] = arc_v
        variables['arc_e'] = arc_e
    
    logger.info(f"MISOCP model created: {N} buses, {E} lines, {len(constraints)} constraints")
    
    return problem, variables, bus_map, line_list


def build_convex_problem(net, toggles, logger):
    if logger is None:
        logger =logging.getLogger(__name__)

     # Default toggles
    if toggles is None:
        toggles = {"include_voltage_drop_constraint": True, 
           "include_voltage_bounds_constraint": True, 
           "include_power_balance_constraint": True, 
           "include_radiality_constraints": True, 
           "use_spanning_tree_radiality": False, 
           "use_root_flow": True, 
           "use_parent_child_radiality": False, 
           "include_switch_penalty": True, 
           "allow_load_shed": True, 
           "all_lines_are_switches": True, "include_cone_constraint": True}
    
    
    print(f"Using toggles: {toggles}")
    # Map buses and lines to sequential indices
    bus_list = list(net.bus.index)
    bus_map  = {b: i for i, b in enumerate(bus_list)}
    line_list = list(net.line.index)

    N = len(bus_list) 
    E = len(line_list)  

    # Extract line parameters
    from_bus_indices = np.array([bus_map[net.line.at[l, 'from_bus']] for l in line_list], dtype=int)
    to_bus_indices = np.array([bus_map[net.line.at[l, 'to_bus']] for l in line_list], dtype=int)
    r_pu     = np.array([net.line.at[l, 'r_pu'] for l in line_list])
    x_pu     = np.array([net.line.at[l, 'x_pu'] for l in line_list])

    # Power injections (p.u.)
    p_inj = np.zeros(N)
    q_inj = np.zeros(N)
    S_base_MVA = 1.0

    for b in bus_list:
        i = bus_map[b]
        p_mw_val = (net.gen[net.gen.bus == b].p_mw.sum() - 
                    net.load[net.load.bus == b].p_mw.sum())
        q_mvar_val = (get_reactive_injection(net.gen, b) -
                      get_reactive_injection(net.load, b))
        p_inj[i] = p_mw_val / S_base_MVA
        q_inj[i] = q_mvar_val / S_base_MVA


    # Big-M values for power flows
    bigM_flow = np.zeros(E)
    for idx, l_pp_idx in enumerate(line_list):
        row = net.line.loc[l_pp_idx]
        if 'max_i_ka' in row and pd.notna(row.max_i_ka) and row.max_i_ka > 0:
            if row.from_bus in net.bus.index:
                v_ln_kv = net.bus.vn_kv[row.from_bus]
                s_max_mva_thermal = np.sqrt(3) * v_ln_kv * row.max_i_ka
                bigM_flow[idx] = s_max_mva_thermal / S_base_MVA
            else:
                bigM_flow[idx] = np.sum(np.abs(p_inj)) if np.sum(np.abs(p_inj)) > 0 else 10.0
        else:
            bigM_flow[idx] = np.sum(np.abs(p_inj)) if np.sum(np.abs(p_inj)) > 0 else 10.0
    
    # Voltage bounds
    v_upper, v_lower = 1.10, 0.90
    bigM_v = (v_upper**2 - v_lower**2) * 1.5

    # Identify substations
    substation_cvx_indices = [bus_map[b_pp_idx] for b_pp_idx in net.ext_grid.bus if b_pp_idx in bus_map]
    
    y0 = np.ones(E) 
    for idx, l_pp_idx in enumerate(line_list):
        sw = net.switch[(net.switch.et == 'l') & (net.switch.element == l_pp_idx)]
        if not sw.empty:
            y0[idx] = 1 if sw.closed.all() else 0


    # continious Variables
    v_sq = cp.Variable(N, name='v_sq', nonneg=True)  # squared voltages
    p_flow = cp.Variable(E, name='p_flow')  # active flows
    q_flow = cp.Variable(E, name='q_flow')  # reactive flows
    I_sq = cp.Variable(E, name='I_sq', nonneg=True)  # squared current magnitudes
    y_line = cp.Variable(E, name='y_line', )  # y_ij: line status
    z_bus = cp.Variable(N, name='z_bus', )  # z_i: bus energization
    
    constraints = []

    constraints += [0 <= y_line,
                    y_line <= 1,
                    0 <= z_bus,
                    z_bus <= 1,]

  
    # SCF variables for radiality
    if toggles.get("include_radiality_constraints") and toggles.get("use_root_flow"):
        # Build directed arcs for SCF
        arcs = [(u, v, e) for e, (u, v) in enumerate(zip(from_bus_indices, to_bus_indices))]
        arcs += [(v, u, e) for e, (u, v) in enumerate(zip(from_bus_indices, to_bus_indices))]
        A = len(arcs)
        
        arc_u = np.array([u for u, v, e in arcs], dtype=int)
        arc_v = np.array([v for u, v, e in arcs], dtype=int)
        arc_e = np.array([e for u, v, e in arcs], dtype=int)
        
        # f_a: commodity flow variables
        f_flow = cp.Variable(A, name='f_flow', nonneg=True)

        flow_dir = cp.Variable(E, name='flow_dir', nonneg=True,)
        constraints.extend([0 <= flow_dir
                            , flow_dir <= 1])

    
    # ----------------------------------------------------------------------------
    # Create constraints
    # ----------------------------------------------------------------------------

    # Constraints list

    # 1. Substations always energized with fixed voltage
    for i_cvx_idx in substation_cvx_indices:
        constraints.append(z_bus[i_cvx_idx] == 1)
        sub_pp_idx = bus_list[i_cvx_idx]
        sub_vm_pu = net.ext_grid[net.ext_grid.bus == sub_pp_idx].vm_pu.iloc[0]
        constraints.append(v_sq[i_cvx_idx] == sub_vm_pu**2)
    
    # 2. Load shedding constraints
    if not toggles.get("allow_load_shed", False):
        for i_cvx_idx in range(N):
            if i_cvx_idx not in substation_cvx_indices:
                constraints.append(z_bus[i_cvx_idx] == 1)
    
    # 3. Voltage bounds
    if toggles.get("include_voltage_bounds_constraint", True):
        for i_cvx_idx in range(N):
            if i_cvx_idx not in substation_cvx_indices:
                # Upper bound: V_i ≤ V_max² + M_v(1-z_i)
                constraints.append(
                    v_sq[i_cvx_idx] <= v_upper**2 + bigM_v * (1 - z_bus[i_cvx_idx])
                )
                # Lower bound: V_i ≥ V_min² * z_i
                constraints.append(
                    v_sq[i_cvx_idx] >= v_lower**2 * z_bus[i_cvx_idx]
                )
    
    # 4. Power flow bounds (Big-M formulation)
    for e_cvx_idx in range(E):
        M = bigM_flow[e_cvx_idx]
        epsilon = 0.01
        constraints.extend([
            p_flow[e_cvx_idx] <= M * y_line[e_cvx_idx],
            p_flow[e_cvx_idx] >= -M * y_line[e_cvx_idx],
            q_flow[e_cvx_idx] <= M * y_line[e_cvx_idx],
            q_flow[e_cvx_idx] >= -M * y_line[e_cvx_idx],
            I_sq[e_cvx_idx] <= M**2 * y_line[e_cvx_idx], 
        ])
    
    # 5. SOCP cone constraints: P² + Q² ≤ V_i * I
    if toggles.get("include_cone_constraint", True):
        for e_cvx_idx in range(E):
            u_idx = from_bus_indices[e_cvx_idx]

            M = bigM_flow[e_cvx_idx]
            constraints.append(
                cp.norm(cp.vstack([
                    2 * p_flow[e_cvx_idx], 
                    2 * q_flow[e_cvx_idx], 
                    v_sq[u_idx] - I_sq[e_cvx_idx]
                ]), 2) <= v_sq[u_idx] + I_sq[e_cvx_idx] + 2*M**2 * (1 - y_line[e_cvx_idx])
            )
    
    # 6. Voltage drop constraints
    if toggles.get("include_voltage_drop_constraint", True):
        for e_cvx_idx in range(E):
            u_idx = from_bus_indices[e_cvx_idx]
            v_idx = to_bus_indices[e_cvx_idx]
            R_l = r_pu[e_cvx_idx]
            X_l = x_pu[e_cvx_idx]
            
            # V_j = V_i - 2(R*P + X*Q) + (R²+X²)*I
            voltage_drop = (v_sq[v_idx] - v_sq[u_idx] + 
                          2 * (R_l * p_flow[e_cvx_idx] + X_l * q_flow[e_cvx_idx]) - 
                          (R_l**2 + X_l**2) * I_sq[e_cvx_idx])
            
            constraints.extend([
                voltage_drop <= bigM_v * (1 - y_line[e_cvx_idx]),
                voltage_drop >= -bigM_v * (1 - y_line[e_cvx_idx])
            ])
    
    # 7. Line-bus linking constraints
    for e_cvx_idx in range(E):
        constraints.extend([
            y_line[e_cvx_idx] <= z_bus[from_bus_indices[e_cvx_idx]],
            y_line[e_cvx_idx] <= z_bus[to_bus_indices[e_cvx_idx]]
        ])
        constraints.extend([
            z_bus[from_bus_indices[e_cvx_idx]] >= y_line[e_cvx_idx],
            z_bus[to_bus_indices[e_cvx_idx]] >= y_line[e_cvx_idx]
        ])
    
    # 8. Bus connectivity constraints
    if toggles.get("include_radiality_constraints", True):
        for i_cvx_idx in range(N):
            if i_cvx_idx not in substation_cvx_indices:
                incident_lines = [e for e in range(E) if 
                                from_bus_indices[e] == i_cvx_idx or 
                                to_bus_indices[e] == i_cvx_idx]
                if incident_lines:
                    constraints.append(
                        cp.sum([y_line[e] for e in incident_lines]) >= z_bus[i_cvx_idx]
                    )
    
    # 9. Power balance constraints
    if toggles.get("include_power_balance_constraint", True):
        for i_cvx_idx in range(N):
            if i_cvx_idx not in substation_cvx_indices:
                # Active power balance
                p_out_lines = [e for e in range(E) if from_bus_indices[e] == i_cvx_idx]
                p_in_lines = [e for e in range(E) if to_bus_indices[e] == i_cvx_idx]
                
                p_out = cp.sum([p_flow[e] for e in p_out_lines]) if p_out_lines else 0
                p_in = cp.sum([p_flow[e] for e in p_in_lines]) if p_in_lines else 0
                
                # constraints.append(p_out - p_in == p_inj[i_cvx_idx] * z_bus[i_cvx_idx])
                
                # Reactive power balance
                q_out = cp.sum([q_flow[e] for e in p_out_lines]) if p_out_lines else 0
                q_in = cp.sum([q_flow[e] for e in p_in_lines]) if p_in_lines else 0
                
                # constraints.append(q_out - q_in == q_inj[i_cvx_idx] * z_bus[i_cvx_idx])

                if p_inj[i_cvx_idx] < 0:
                    constraints.append(p_out - p_in >= p_inj[i_cvx_idx] * z_bus[i_cvx_idx])
                    constraints.append(p_out - p_in <= 0)  # Can't generate if load bus
                else:
                    constraints.append(p_out - p_in == p_inj[i_cvx_idx] * z_bus[i_cvx_idx])

                if q_inj[i_cvx_idx] < 0:
                    constraints.append(q_out - q_in >= q_inj[i_cvx_idx] * z_bus[i_cvx_idx])
                    constraints.append(q_out - q_in <= 0)  # Can't generate if load bus
                else:
                    constraints.append(q_out - q_in == q_inj[i_cvx_idx] * z_bus[i_cvx_idx])

    # 10. SCF radiality constraints
    if toggles.get("include_radiality_constraints", True) and toggles.get("use_root_flow", True):
        print("Using SCF radiality constraints")
        # SCF capacity constraints
        M_scf = N - 1
        # SCF conservation
        root_idx = substation_cvx_indices[0] if substation_cvx_indices else 0
        
        for i_cvx_idx in range(N):
            in_arcs = [a for a in range(A) if arc_v[a] == i_cvx_idx]
            out_arcs = [a for a in range(A) if arc_u[a] == i_cvx_idx]
            
            f_in = cp.sum([f_flow[a] for a in in_arcs]) if in_arcs else 0
            f_out = cp.sum([f_flow[a] for a in out_arcs]) if out_arcs else 0
            
            if i_cvx_idx == root_idx:
                constraints.append(f_out - f_in == cp.sum(z_bus) - 1)
            else:
                constraints.append(f_in - f_out == z_bus[i_cvx_idx])
        for e_idx in range(E):
            forward_arc = e_idx
            reverse_arc = e_idx + E
            
            constraints.append(f_flow[forward_arc] + f_flow[reverse_arc] <= M_scf * y_line[e_idx])

        constraints.append(cp.sum(y_line) == cp.sum(z_bus) - 1)
    # ==================================
    # Objective Function
    # ==================================

    # Objective function
    loss_objective = cp.sum(cp.multiply(r_pu, I_sq))
    
    # Add switch penalty if enabled
    if toggles.get("include_switch_penalty", True):
        switch_penalty = 0.0001  
        switch_changes = cp.sum(cp.abs(y_line  - y0))
        loss_objective += switch_penalty * switch_changes
    
    # Add load shedding penalty if enabled
    if toggles.get("allow_load_shed", False):
        load_shed_penalty = 100  
        load_buses = [i for i in range(N) if i not in substation_cvx_indices and p_inj[i] < 0]
        shed_penalty = cp.sum([cp.abs(p_inj[i]) * (1 - z_bus[i]) for i in load_buses])
        loss_objective += load_shed_penalty * shed_penalty
    
    # Create the optimization problem
    problem = cp.Problem(cp.Minimize(loss_objective), constraints)

    # Store variables
    variables = {
        'v_sq': v_sq,
        'p_flow': p_flow,
        'q_flow': q_flow,
        'I_sq': I_sq,
        'y_line': y_line,
        'z_bus': z_bus,
    }
    print(f"Variables: {variables.keys()}")
    if toggles.get("include_radiality_constraints") and toggles.get("use_root_flow"):
        variables['f_flow'] = f_flow
        variables['flow_dir'] = flow_dir
        variables['arc_u'] = arc_u
        variables['arc_v'] = arc_v
        variables['arc_e'] = arc_e
    
    logger.info(f"MISOCP model created: {N} buses, {E} lines, {len(constraints)} constraints")
    
    return problem, variables, bus_map, line_list


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
    


