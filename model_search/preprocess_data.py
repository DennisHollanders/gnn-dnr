from pyparsing import line
import torch 
import networkx as nx
import pandas as pd
import numpy  as np

def calculate_conductance_matrix(graph):
    num_nodes = graph.number_of_nodes()
    
    # Initialize sparse matrix indices and values
    indices = []
    values = []
    
    # Add entries for each edge
    for u, v, data in graph.edges(data=True):
        # Get resistance (R) from edge attributes, default to 0.01 if not available
        r_value = data.get('R', 0.01)
        if r_value <= 0:
            r_value = 1e-6  # Avoid division by zero
        
        # Conductance is 1/R
        conductance = 1.0 / r_value
        
        # Add entries for both directions (symmetric matrix)
        indices.append([u, v])
        indices.append([v, u])
        values.extend([conductance, conductance])
        
    
    if not indices:  # If no edges were processed
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float),
            (num_nodes, num_nodes)
        )
    
    # Convert lists to tensors
    indices_tensor = torch.tensor(indices, dtype=torch.long).t()
    values_tensor = torch.tensor(values, dtype=torch.float)
    
    # Create sparse tensor
    conductance_matrix = torch.sparse_coo_tensor(
        indices_tensor, values_tensor, (num_nodes, num_nodes)
    )
    
    diagonal_indices = []
    diagonal_values = []
    for node in range(num_nodes):
        mask = indices_tensor[0] == node
        outgoing_sum = values_tensor[mask].sum().item()
        diagonal_indices.append([node, node])
        diagonal_values.append(-outgoing_sum)
    
    diag_indices_tensor = torch.tensor(diagonal_indices, dtype=torch.long).t()
    diag_values_tensor = torch.tensor(diagonal_values, dtype=torch.float)
    diag_matrix = torch.sparse_coo_tensor(diag_indices_tensor, diag_values_tensor, (num_nodes, num_nodes))
    
    return conductance_matrix + diag_matrix


def calculate_laplacian_matrix(graph):
    # Get adjacency matrix as a sparse tensor
    adj_matrix = nx.to_scipy_sparse_array(graph, weight=None)
    adj_tensor = torch.tensor(adj_matrix.todense(), dtype=torch.float)
    
    # Calculate degree matrix
    degrees = adj_tensor.sum(dim=1)
    n = graph.number_of_nodes()
    
    # Create sparse degree matrix
    degree_indices = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
    degree_matrix = torch.sparse_coo_tensor(degree_indices, degrees, (n, n))
    
    # Convert adj_tensor to sparse
    adj_indices = torch.nonzero(adj_tensor).t()
    adj_values = adj_tensor[adj_indices[0], adj_indices[1]]
    adj_sparse = torch.sparse_coo_tensor(adj_indices, adj_values, (n, n))
    
    # Laplacian = D - A
    # For sparse tensors, we negate the values of adj_sparse
    neg_adj_values = -adj_values
    neg_adj_sparse = torch.sparse_coo_tensor(adj_indices, neg_adj_values, (n, n))
    
    # Add the sparse matrices
    laplacian = degree_matrix + neg_adj_sparse
    
    return laplacian


def calculate_admittance_matrix(graph):
    num_nodes = graph.number_of_nodes()
    
    # Initialize sparse matrix indices and values
    indices = []
    values = []
    
    # Add entries for each edge
    for u, v, data in graph.edges(data=True):
        # Get resistance (R) and reactance (X) from edge attributes
        r_value = data.get('R', 0.01)
        x_value = data.get('X', 0.01)
        
        # Calculate admittance (inverse of impedance)
        z = complex(r_value, x_value)
        if abs(z) <= 1e-10:
            z = complex(1e-6, 1e-6)  # Avoid division by zero
        
        y = 1.0 / z
        
        # Add entries for both directions (symmetric matrix)
        # Store real and imaginary parts separately
        indices.append([u, v])
        indices.append([v, u])
        values.extend([y.real, y.real])  # Only storing real part for now
        
        # We'll handle diagonal entries separately
    
    if not indices:  # If no edges were processed
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float),
            (num_nodes, num_nodes)
        )
    
    # Convert lists to tensors
    indices_tensor = torch.tensor(indices, dtype=torch.long).t()
    values_tensor = torch.tensor(values, dtype=torch.float)
    
    # Create sparse tensor
    admittance_matrix = torch.sparse_coo_tensor(
        indices_tensor, values_tensor, (num_nodes, num_nodes)
    )
    
    # Calculate diagonal entries (negative sum of off-diagonal entries in each row)
    diagonal_values = []
    diagonal_indices = []
    
    for node in range(num_nodes):
        # Get all values where this node is the source
        mask = indices_tensor[0] == node
        outgoing_sum = values_tensor[mask].sum().item()
        
        # Add diagonal entry
        diagonal_indices.append([node, node])
        diagonal_values.append(-outgoing_sum)
    
    # Add diagonal entries to sparse matrix
    if diagonal_indices:
        diag_indices_tensor = torch.tensor(diagonal_indices, dtype=torch.long).t()
        diag_values_tensor = torch.tensor(diagonal_values, dtype=torch.float)
        
        # Create sparse tensor for diagonal
        diag_matrix = torch.sparse_coo_tensor(
            diag_indices_tensor, diag_values_tensor, (num_nodes, num_nodes)
        )
        
        # Add to the admittance matrix
        admittance_matrix = admittance_matrix + diag_matrix
    
    return admittance_matrix

def calculate_switch_matrix(nx_graph):
    """
    Calculate switch matrix as a sparse tensor where entries
    indicate switch connections between nodes.
    """
    num_nodes = nx_graph.number_of_nodes()
    
    # Initialize sparse matrix indices and values
    indices = []
    values = []
    
    # Add entries for each edge
    for u, v, data in nx_graph.edges(data=True):
        switch_state = data.get('switch_state', 0)
        
        # Only add edges that are switches (switch_state > 0)
        if switch_state > 0:
            # Add entries for both directions (symmetric matrix)
            indices.append([u, v])
            indices.append([v, u])
            values.extend([1.0, 1.0])  # 1.0 indicates a switch exists
    
    if not indices:  # If no switches were found
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float),
            (num_nodes, num_nodes)
        )
    
    # Convert lists to tensors
    indices_tensor = torch.tensor(indices, dtype=torch.long).t()
    values_tensor = torch.tensor(values, dtype=torch.float)
    
    # Create sparse tensor
    switch_matrix = torch.sparse_coo_tensor(
        indices_tensor, values_tensor, (num_nodes, num_nodes)
    )
    
    return switch_matrix

def calculate_adjacency_matrix(graph):
    """
    Calculate adjacency matrix as a sparse tensor.
    """
    num_nodes = graph.number_of_nodes()
    indices = []
    values = []
    
    # Add entries for each edge
    for u, v in graph.edges():
        indices.append([u, v])
        values.append(1.0)
        
        if not graph.is_directed():
            indices.append([v, u])
            values.append(1.0)
    
    if not indices:  
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float),
            (num_nodes, num_nodes)
        )
    
    # Convert lists to tensors
    indices_tensor = torch.tensor(indices, dtype=torch.long).t()
    values_tensor = torch.tensor(values, dtype=torch.float)
    
    # Create sparse tensor
    adjacency_matrix = torch.sparse_coo_tensor(
        indices_tensor, values_tensor, (num_nodes, num_nodes)
    )
    
    return adjacency_matrix


def cxv_features(pp_net):
    bus_ids = pp_net.bus.index.to_numpy(dtype=int)
    id2row  = {bid: i for i, bid in enumerate(bus_ids)}
    N = len(bus_ids)


    line_df = pp_net.line
    E = len(line_df)

    required_line_cols = ['from_bus', 'to_bus', 'r_ohm_per_km', 'x_ohm_per_km'] # Add any other essential cols
    missing_cols = [col for col in required_line_cols if col not in line_df.columns]
    if missing_cols:
        raise AttributeError(
            f"Missing required columns in pp_net.line: {missing_cols}. "
            f"Available columns: {line_df.columns.tolist()}"
        )

    from_idx = torch.from_numpy(
        line_df.from_bus.map(id2row).to_numpy(dtype=np.int64)
    )
    to_idx   = torch.from_numpy(
        line_df.to_bus.map(id2row).to_numpy(dtype=np.int64)
    )

    r_pu_values = line_df.r_ohm_per_km.values.astype(np.float32) * line_df.length_km.values.astype(np.float32)
    x_pu_values = line_df.x_ohm_per_km.values.astype(np.float32) * line_df.length_km.values.astype(np.float32)
    r_pu = torch.from_numpy(r_pu_values)
    x_pu = torch.from_numpy(x_pu_values)

    p_inj_np = np.zeros(N, dtype=np.float32)
    q_inj_np = np.zeros(N, dtype=np.float32)
    S_base = 1.0


    if 'gen' in pp_net and not pp_net.gen.empty:
        gen_grouped = pp_net.gen.groupby('bus')[['p_mw', 'q_mvar']].sum()
    else:
        gen_grouped = pd.DataFrame(columns=['p_mw', 'q_mvar'])

    if 'load' in pp_net and not pp_net.load.empty:
        load_grouped = pp_net.load.groupby('bus')[['p_mw', 'q_mvar']].sum()
    else:
        load_grouped = pd.DataFrame(columns=['p_mw', 'q_mvar'])

    for bid, i in id2row.items():
        p_gen = gen_grouped.at[bid, 'p_mw'] if bid in gen_grouped.index else 0
        q_gen = gen_grouped.at[bid, 'q_mvar'] if bid in gen_grouped.index else 0
        p_load = load_grouped.at[bid, 'p_mw'] if bid in load_grouped.index else 0
        q_load = load_grouped.at[bid, 'q_mvar'] if bid in load_grouped.index else 0
        
        p_inj_np[i] = (p_gen - p_load) / S_base
        q_inj_np[i] = (q_gen - q_load) / S_base
        
    p_inj = torch.from_numpy(p_inj_np)
    q_inj = torch.from_numpy(q_inj_np)

   
    bigM_flow_list = [] 
    total_load_abs_sum = np.abs(p_inj_np).sum() 
    

    if not ('vn_kv' in pp_net.bus.columns and pp_net.bus.vn_kv.notna().all()):
         print("Warning: 'vn_kv' column missing or has NaN values in pp_net.bus. bigM_flow might be inaccurate.")

    for _, row in line_df.iterrows():
        M_val = total_load_abs_sum if total_load_abs_sum > 0 else 10.0 
        if pd.notna(row.get('max_i_ka')) and row.max_i_ka > 0:
            if row.from_bus in pp_net.bus.index and 'vn_kv' in pp_net.bus.columns:
                v_kv = pp_net.bus.vn_kv.at[row.from_bus]
                if pd.notna(v_kv):
                    s_max = np.sqrt(3) * v_kv * row.max_i_ka
                    M_val = s_max / S_base
        bigM_flow_list.append(M_val)
    bigM_flow = torch.tensor(bigM_flow_list, dtype=torch.float32)


    vmin, vmax = 0.9, 1.10 
    bigM_v_val = float((vmax**2 - vmin**2) * 1.5)

    sub_idx_list = [] 
    if 'ext_grid' in pp_net and not pp_net.ext_grid.empty:
        sub_idx_list = [id2row[bid] for bid in pp_net.ext_grid.bus if bid in id2row]
    sub_idx = torch.tensor(sub_idx_list, dtype=torch.long) 

    y0_np = np.ones(E, dtype=np.float32) 
    if 'switch' in pp_net and not pp_net.switch.empty:
        if all(col in pp_net.switch.columns for col in ['et', 'element', 'closed']):
            line_switches = pp_net.switch[pp_net.switch.et == 'l']
            line_original_indices = line_df.index.to_numpy()
            line_pos_map = {orig_idx: pos for pos, orig_idx in enumerate(line_original_indices)}

            for _, r_sw in line_switches.iterrows():
                line_orig_idx = r_sw.element
                if line_orig_idx in line_pos_map:
                    pos = line_pos_map[line_orig_idx]
                    y0_np[pos] = float(r_sw.closed)
                else:
                    print(f"Warning: Switch element {line_orig_idx} not found in line table. Ignoring for y0.")
        else:
            print("Warning: Switch table missing required columns (et, element, closed). Using default y0.")
            
    y0 = torch.from_numpy(y0_np)


    return {
        'N': N, 'E': E,
        'from_idx': from_idx, 'to_idx': to_idx,
        'r_pu': r_pu, 'x_pu': x_pu,
        'p_inj': p_inj, 'q_inj': q_inj,
        'bigM_flow': bigM_flow, 'bigM_v': bigM_v_val, 
        'sub_idx': sub_idx, 
        'y0': y0,

    }