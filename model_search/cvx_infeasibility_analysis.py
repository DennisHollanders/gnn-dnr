import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def analyze_cvx_infeasibility(data):
    """
    Analyze what's causing CVX infeasibility in your current setup.
    """
    print("\n🔍 CVX INFEASIBILITY ANALYSIS")
    print("="*60)
    
    # Extract key information from batch
    actual_nodes = data.x.shape[0]  # 108 nodes from your log
    actual_edges = data.edge_index.shape[1]  # 112 edges from your log
    
    # CVX dimensions from padding
    cvx_N = data.cvx_N.item()  # Should be 1 based on your log
    cvx_E = data.cvx_E.item()  # Should be 1 based on your log
    
    print(f"📊 DIMENSION ANALYSIS:")
    print(f"   Actual graph: {actual_nodes} nodes, {actual_edges} edges")
    print(f"   CVX expects: {cvx_N} nodes, {cvx_E} edges")
    print(f"   🚨 MASSIVE MISMATCH: Graph is 100x larger than CVX thinks!")
    
    # Check constraint matrices
    if hasattr(data, 'cvx_A_from') and hasattr(data, 'cvx_A_to'):
        A_from = data.cvx_A_from  # Shape should be [1, 147, 140]
        A_to = data.cvx_A_to      # Shape should be [1, 147, 140]
        
        print(f"🔗 CONSTRAINT MATRICES:")
        print(f"   A_from shape: {A_from.shape}")
        print(f"   A_to shape: {A_to.shape}")
        
        if len(A_from.shape) == 3:
            batch_size, matrix_E, matrix_N = A_from.shape
            print(f"   Matrix dimensions: {matrix_E} edges × {matrix_N} nodes")
            
            # This is the CORE PROBLEM:
            # CVX thinks it has 1 node and 1 edge
            # But constraint matrices are 147×140
            # Solver tries to optimize over 1 variable with 147×140 constraints
            print(f"   🚨 CONSTRAINT DIMENSION MISMATCH:")
            print(f"      CVX variables: {cvx_E} edge vars, {cvx_N} node vars")
            print(f"      Constraint matrix: {matrix_E}×{matrix_N}")
            print(f"      This creates {matrix_E * matrix_N} constraints for {cvx_E + cvx_N} variables!")
    
    # Check injections
    if hasattr(data, 'cvx_p_inj'):
        p_inj = data.cvx_p_inj.squeeze()
        q_inj = data.cvx_q_inj.squeeze()
        
        print(f"⚡ INJECTION ANALYSIS:")
        print(f"   P injection shape: {p_inj.shape}")
        print(f"   Expected for CVX_N={cvx_N}: shape should be [{cvx_N}]")
        
        if p_inj.shape[0] != cvx_N:
            print(f"   🚨 INJECTION SIZE MISMATCH:")
            print(f"      Have {p_inj.shape[0]} injection values")
            print(f"      CVX expects {cvx_N} injection values")
        
        # Check if injections sum to zero (power balance)
        p_sum = p_inj.sum().item()
        q_sum = q_inj.sum().item()
        print(f"   Power balance: P_sum={p_sum:.6f}, Q_sum={q_sum:.6f}")
        
        if abs(p_sum) > 1e-3:
            print(f"   ⚠️  Power imbalance detected: {p_sum:.6f}")
    
    # Check bigM values
    if hasattr(data, 'cvx_bigM_flow'):
        bigM_flow = data.cvx_bigM_flow.squeeze()
        bigM_v = data.cvx_bigM_v.item()
        
        print(f"🔧 BIG-M ANALYSIS:")
        print(f"   BigM flow shape: {bigM_flow.shape}")
        print(f"   Expected for CVX_E={cvx_E}: shape should be [{cvx_E}]")
        print(f"   BigM voltage: {bigM_v}")
        
        if bigM_flow.shape[0] != cvx_E:
            print(f"   🚨 BIG-M SIZE MISMATCH:")
            print(f"      Have {bigM_flow.shape[0]} bigM values")
            print(f"      CVX expects {cvx_E} bigM values")
    
    # Identify the ROOT CAUSE
    print(f"\n🎯 ROOT CAUSE IDENTIFIED:")
    print(f"   The CVX layer is receiving:")
    print(f"   1. Problem size: {cvx_N} nodes, {cvx_E} edges (WRONG)")
    print(f"   2. Constraint matrices: {A_from.shape[1]}×{A_from.shape[2]} (CORRECT)")
    print(f"   3. This creates an over-constrained infeasible system")
    
    return {
        'actual_N': actual_nodes,
        'actual_E': actual_edges,
        'cvx_N': cvx_N,
        'cvx_E': cvx_E,
        'dimension_mismatch': (cvx_N != actual_nodes) or (cvx_E != actual_edges)
    }


def fix_cvx_n_e_values(data):
    """
    Fix the core issue: CVX_N and CVX_E are wrong after batching.
    """
    print("\n🔧 FIXING CVX_N AND CVX_E VALUES")
    print("-"*40)
    
    # Get actual dimensions from the batched graph
    actual_nodes = data.x.shape[0]
    actual_edges = data.edge_index.shape[1]
    
    print(f"Correcting CVX dimensions:")
    print(f"   Before: N={data.cvx_N.item()}, E={data.cvx_E.item()}")
    print(f"   After:  N={actual_nodes}, E={actual_edges}")
    
    # Fix the CVX dimensions
    data.cvx_N = torch.tensor([actual_nodes], dtype=torch.long)
    data.cvx_E = torch.tensor([actual_edges], dtype=torch.long)
    
    # Now check if constraint matrices match
    if hasattr(data, 'cvx_A_from'):
        matrix_E, matrix_N = data.cvx_A_from.shape[1], data.cvx_A_from.shape[2]
        
        if matrix_E < actual_edges or matrix_N < actual_nodes:
            print(f"⚠️  Constraint matrices too small:")
            print(f"     Matrix: {matrix_E}×{matrix_N}")
            print(f"     Needed: {actual_edges}×{actual_nodes}")
            print(f"     Need to recreate constraint matrices...")
            
            # Recreate constraint matrices with correct size
            return recreate_constraint_matrices(data, actual_nodes, actual_edges)
    
    return data


def recreate_constraint_matrices(data, N, E):
    """
    Recreate constraint matrices based on actual graph structure.
    """
    print(f"🔨 Recreating constraint matrices for N={N}, E={E}")
    
    # Get edge structure from the actual batched graph
    edge_index = data.edge_index  # [2, E]
    
    # Create incidence matrices
    A_from = torch.zeros((1, E, N), dtype=torch.float32)
    A_to = torch.zeros((1, E, N), dtype=torch.float32)
    
    for e in range(E):
        from_node = edge_index[0, e].item()
        to_node = edge_index[1, e].item()
        
        # Ensure indices are within bounds
        if from_node < N and to_node < N:
            A_from[0, e, from_node] = 1.0
            A_to[0, e, to_node] = 1.0
        else:
            print(f"⚠️  Edge {e}: nodes ({from_node}, {to_node}) exceed N={N}")
    
    data.cvx_A_from = A_from
    data.cvx_A_to = A_to
    
    # Update masks to match actual dimensions
    node_mask = torch.zeros(N, dtype=torch.bool)
    node_mask[:N] = True  # All nodes are active
    data.cvx_node_mask = node_mask.unsqueeze(0)
    
    edge_mask = torch.zeros(E, dtype=torch.bool)
    edge_mask[:E] = True  # All edges are active
    data.cvx_edge_mask = edge_mask.unsqueeze(0)
    
    # Resize other tensors if needed
    tensor_attrs = [
        ('cvx_r_pu', E), ('cvx_x_pu', E), ('cvx_bigM_flow', E), ('cvx_y0', E),
        ('cvx_p_inj', N), ('cvx_q_inj', N)
    ]
    
    for attr_name, expected_size in tensor_attrs:
        if hasattr(data, attr_name):
            tensor = getattr(data, attr_name)
            if tensor.shape[1] < expected_size:
                # Pad tensor to required size
                current_size = tensor.shape[1]
                padding_size = expected_size - current_size
                padding = torch.zeros((1, padding_size), dtype=tensor.dtype)
                new_tensor = torch.cat([tensor, padding], dim=1)
                setattr(data, attr_name, new_tensor)
                print(f"   Padded {attr_name}: {tensor.shape} → {new_tensor.shape}")
            elif tensor.shape[1] > expected_size:
                # Truncate tensor
                new_tensor = tensor[:, :expected_size]
                setattr(data, attr_name, new_tensor)
                print(f"   Truncated {attr_name}: {tensor.shape} → {new_tensor.shape}")
    
    print("✅ Constraint matrices recreated")
    return data


def validate_cvx_problem_setup(data):
    """
    Validate that the CVX problem setup is mathematically sound.
    """
    print("\n✅ VALIDATING CVX PROBLEM SETUP")
    print("-"*40)
    
    N = data.cvx_N.item()
    E = data.cvx_E.item()
    
    # Basic dimension checks
    print(f"Problem dimensions: N={N}, E={E}")
    
    # Check power balance
    p_inj = data.cvx_p_inj.squeeze()[:N]  # Only consider active nodes
    q_inj = data.cvx_q_inj.squeeze()[:N]
    
    p_balance = p_inj.sum().item()
    q_balance = q_inj.sum().item()
    
    print(f"Power balance:")
    print(f"   P total: {p_balance:.6f} (should be ≈ 0)")
    print(f"   Q total: {q_balance:.6f} (should be ≈ 0)")
    
    if abs(p_balance) > 1e-3:
        print(f"   ⚠️  Significant power imbalance: {p_balance:.6f}")
        
        # Check if there's a substation
        if hasattr(data, 'cvx_sub_idx') and data.cvx_sub_idx.numel() > 0:
            sub_indices = data.cvx_sub_idx.squeeze()
            print(f"   Substation indices: {sub_indices}")
            print(f"   Consider: substation should balance total load")
    
    # Check connectivity
    edge_index = data.edge_index
    num_connected_nodes = len(torch.unique(edge_index))
    print(f"Graph connectivity:")
    print(f"   Nodes in edges: {num_connected_nodes}/{N}")
    
    if num_connected_nodes < N:
        isolated_nodes = N - num_connected_nodes
        print(f"   ⚠️  {isolated_nodes} isolated nodes detected")
    
    # Check for minimum spanning tree feasibility
    if E < N - 1:
        print(f"   ⚠️  Too few edges for connectivity: {E} < {N-1}")
    
    # Check bigM values
    bigM_flow = data.cvx_bigM_flow.squeeze()[:E]
    bigM_v = data.cvx_bigM_v.item()
    
    print(f"BigM constraints:")
    print(f"   Flow BigM: min={bigM_flow.min():.4f}, max={bigM_flow.max():.4f}")
    print(f"   Voltage BigM: {bigM_v:.4f}")
    
    if bigM_flow.min() < 1e-6:
        print(f"   ⚠️  Very small BigM flow values may cause numerical issues")
    
    return {
        'power_balanced': abs(p_balance) < 1e-3,
        'fully_connected': num_connected_nodes == N,
        'sufficient_edges': E >= N - 1,
        'reasonable_bigM': bigM_flow.min() > 1e-6
    }


# Main fix function to integrate into your code
def fix_cvx_infeasibility(data):
    """
    Complete fix for CVX infeasibility issues.
    Call this on each batch before passing to CVX layer.
    """
    print("\n🚨 FIXING CVX INFEASIBILITY")
    print("="*50)
    
    # Step 1: Analyze the problem
    analysis = analyze_cvx_infeasibility(data)
    
    # Step 2: Fix dimension mismatches
    if analysis['dimension_mismatch']:
        print("\n🔧 Fixing dimension mismatches...")
        data = fix_cvx_n_e_values(data)
    
    # Step 3: Validate the setup
    validation = validate_cvx_problem_setup(data)
    
    # Step 4: Report results
    print(f"\n📋 FIX SUMMARY:")
    print(f"   Power balanced: {'✅' if validation['power_balanced'] else '❌'}")
    print(f"   Fully connected: {'✅' if validation['fully_connected'] else '❌'}")
    print(f"   Sufficient edges: {'✅' if validation['sufficient_edges'] else '❌'}")
    print(f"   Reasonable BigM: {'✅' if validation['reasonable_bigM'] else '❌'}")
    
    if all(validation.values()):
        print("✅ CVX problem should now be feasible!")
    else:
        print("⚠️  Some issues remain - may still be infeasible")
    
    return data, validation


# Integration example for your train.py
def process_batch_with_cvx_fix(model, data, criterion, device, **kwargs):
    """
    Modified process_batch that fixes CVX issues before model forward pass.
    """
    if data is None:
        logger.debug("Warning: Data is None, skipping this batch.")
        return torch.tensor(0.0, device=device), {}, {"valid_batch": False}

    data = data.to(device)
    
    # 🔧 FIX CVX INFEASIBILITY BEFORE MODEL CALL
    try:
        data, validation = fix_cvx_infeasibility(data)
        
        if not all(validation.values()):
            logger.warning("CVX validation failed - model may still be infeasible")
    
    except Exception as e:
        logger.error(f"CVX fix failed: {e}")
        return torch.tensor(0.0, device=device), {}, {"valid_batch": False}
    
    # Continue with your normal process_batch logic...
    output = model(data)
    # ... rest of your process_batch function
    
    return output  # Replace with your actual return values