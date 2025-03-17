import torch 
import networkx as nx
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
        
        # Add self-loops with negative sum of conductances for each node
        # We'll handle this after processing all edges
    
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
    
    # Calculate diagonal entries (negative sum of off-diagonal entries in each row)
    # For each node, sum up all outgoing conductances and negate
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
        
        # Add to the conductance matrix
        conductance_matrix = conductance_matrix + diag_matrix
    
    return conductance_matrix


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
    
    # Initialize sparse matrix indices and values
    indices = []
    values = []
    
    # Add entries for each edge
    for u, v in graph.edges():
        # Add entries for both directions (symmetric matrix for undirected graph)
        indices.append([u, v])
        values.append(1.0)
        
        # If the graph is undirected, add the reverse edge as well
        if not graph.is_directed():
            indices.append([v, u])
            values.append(1.0)
    
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
    adjacency_matrix = torch.sparse_coo_tensor(
        indices_tensor, values_tensor, (num_nodes, num_nodes)
    )
    
    return adjacency_matrix