from pyparsing import line
import torch 
import networkx as nx
import pandas as pd
import numpy  as np

def calculate_conductance_matrix(graph):
    num_nodes = graph.number_of_nodes()
    indices = []
    values = []
    
    # Add entries for each edge
    for u, v, data in graph.edges(data=True):
        r_value = data.get('R', 0.01)
        if r_value <= 0:
            r_value = 1e-6 
        conductance = 1.0 / r_value
        
        indices.append([u, v])
        indices.append([v, u])
        values.extend([conductance, conductance])
        
    
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
        r_value = data.get('R', 0.01)
        x_value = data.get('X', 0.01)
        
        # Calculate admittance 
        z = complex(r_value, x_value)
        if abs(z) <= 1e-10:
            z = complex(1e-6, 1e-6)  
        
        y = 1.0 / z
        
        # Store real and imaginary parts separately
        indices.append([u, v])
        indices.append([v, u])
        values.extend([y.real, y.real])  
        
    
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
    admittance_matrix = torch.sparse_coo_tensor(
        indices_tensor, values_tensor, (num_nodes, num_nodes)
    )
    
    # Calculate diagonal entries
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
        
        # Only add edges that are switches 
        if switch_state > 0:
            indices.append([u, v])
            indices.append([v, u])
            values.extend([1.0, 1.0])  
    
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


def cxv_features(pp_net, pf_range=(0.85, 0.95), seed=None):
    # ——— ensure reproducibility if desired ———
    rng = np.random.default_rng(seed)

    # bus lookup
    bus_ids = pp_net.bus.index.to_numpy(dtype=int)
    id2row  = {bid: i for i, bid in enumerate(bus_ids)}
    N = len(bus_ids)

    # line data
    line_df = pp_net.line
    E = len(line_df)
    
    for tbl in ('load', 'gen'):
        df = getattr(pp_net, tbl)
        if 'q_mvar' not in df.columns or df['q_mvar'].isnull().all():
            pf = rng.uniform(pf_range[0], pf_range[1], size=len(df))
            df['q_mvar'] = df['p_mw'].to_numpy() * np.tan(np.arccos(pf))
            setattr(pp_net, tbl, df)

    # map from/to indices
    from_idx = torch.from_numpy(line_df.from_bus.map(id2row).to_numpy(dtype=np.int64))
    to_idx   = torch.from_numpy(line_df.to_bus  .map(id2row).to_numpy(dtype=np.int64))

    # line impedances 
    r_pu = torch.from_numpy((line_df.r_ohm_per_km  * line_df.length_km).to_numpy(dtype=np.float32))
    x_pu = torch.from_numpy((line_df.x_ohm_per_km  * line_df.length_km).to_numpy(dtype=np.float32))

    # net injections
    p_inj_np = np.zeros(N, dtype=np.float32)
    q_inj_np = np.zeros(N, dtype=np.float32)
    S_base = 1.0

    # sum gen and load by bus
    gen_grp  = pp_net.gen .groupby('bus')[['p_mw', 'q_mvar']].sum()
    load_grp = pp_net.load.groupby('bus')[['p_mw', 'q_mvar']].sum()

    for bid, i in id2row.items():
        p_gen = gen_grp .at[bid, 'p_mw']    if bid in gen_grp .index else 0
        q_gen = gen_grp .at[bid, 'q_mvar']  if bid in gen_grp .index else 0
        p_load= load_grp.at[bid, 'p_mw']    if bid in load_grp.index else 0
        q_load= load_grp.at[bid, 'q_mvar']  if bid in load_grp.index else 0
        p_inj_np[i] = (p_gen - p_load) / S_base
        q_inj_np[i] = (q_gen - q_load) / S_base

    # big-M flow
    total_load = np.abs(p_inj_np).sum()
    bigM_flow = torch.tensor([
        (np.sqrt(3) * pp_net.bus.vn_kv.at[row.from_bus] * row.max_i_ka) / S_base
        if pd.notna(row.get('max_i_ka')) and row.max_i_ka > 0 and 'vn_kv' in pp_net.bus.columns
        else (total_load if total_load>0 else 10.0)
        for _, row in line_df.iterrows()
    ], dtype=torch.float32)

    # big-M voltage
    bigM_v = float((1.10**2 - 0.90**2) * 1.5)

    # substations
    sub_idx = torch.tensor([
        id2row[bid] for bid in pp_net.ext_grid.bus if bid in id2row
    ], dtype=torch.long) if 'ext_grid' in pp_net and not pp_net.ext_grid.empty else torch.empty(0, dtype=torch.long)

    # initial switch state y0
    y0 = torch.ones(E, dtype=torch.float32)
    if 'switch' in pp_net and not pp_net.switch.empty:
        sw = pp_net.switch
        if all(c in sw.columns for c in ('et','element','closed')):
            ls = sw[sw.et=='l']
            pos_map = {orig:pos for pos,orig in enumerate(line_df.index)}
            for _, r in ls.iterrows():
                if r.element in pos_map:
                    y0[pos_map[r.element]] = float(r.closed)

    return {
        'N': N, 'E': E,
        'from_idx': from_idx, 'to_idx': to_idx,
        'r_pu': r_pu, 'x_pu': x_pu,
        'p_inj': torch.from_numpy(p_inj_np),
        'q_inj': torch.from_numpy(q_inj_np),
        'bigM_flow': bigM_flow, 'bigM_v': bigM_v,
        'sub_idx': sub_idx, 'y0': y0,
    }