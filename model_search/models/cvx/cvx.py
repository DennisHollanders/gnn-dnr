#
# DPP-compliant CVX layer implementation
#
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import logging
from scipy.sparse import csc_matrix

logger = logging.getLogger(__name__)

def build_cvx_layer(sample_data, args):
    """
    Builds a DPP-compliant CvxpyLayer that maintains feasibility.
    
    Key DPP fixes:
    1. Replace boolean variables with continuous [0,1] variables (relaxation)
    2. Use SOC constraints instead of quadratic constraints
    3. Ensure all constraints are DPP-compliant
    4. Use proper parameter handling for DPP
    """
    # --- 1. Use the actual padded dimensions from the data batch ---
    max_n = sample_data.num_nodes
    max_e = sample_data.edge_index.shape[1]

    logger.info(f"DPP-compliant build_cvx_layer for size: max_nodes={max_n}, max_edges={max_e}")

    # --- Use the pre-padded tensors directly ---
    from_idx_np  = sample_data.cvx_from_idx.cpu().numpy().flatten()
    to_idx_np    = sample_data.cvx_to_idx.cpu().numpy().flatten()
    r_pu_np      = sample_data.cvx_r_pu.cpu().numpy().flatten()
    x_pu_np      = sample_data.cvx_x_pu.cpu().numpy().flatten()
    bigM_flow_np = sample_data.cvx_bigM_flow.cpu().numpy().flatten()
    sub_idx_np   = sample_data.cvx_sub_idx.cpu().numpy().flatten()
    non_sub_mask_np = 1 - sub_idx_np
    
    bigM_v = float(sample_data.cvx_bigM_v[0].item())

    v_lower, v_upper = 0.9, 1.1

    # --- 2. Create sparse incidence matrices for vectorization ---
    rows = np.concatenate([np.arange(max_e), np.arange(max_e)])
    cols = np.concatenate([from_idx_np, to_idx_np])
    data = np.concatenate([-np.ones(max_e), np.ones(max_e)])
    A = csc_matrix((data, (rows, cols)), shape=(max_e, max_n))
    A_from = csc_matrix((np.ones(max_e), (np.arange(max_e), from_idx_np)), shape=(max_e, max_n))
    A_to   = csc_matrix((np.ones(max_e), (np.arange(max_e), to_idx_np)), shape=(max_e, max_n))

    # --- 3. Define CVXPY Variables and Parameters ---
    # DPP FIX 1: Use continuous variables in [0,1] instead of boolean
    # This is a common relaxation that makes the problem DPP-compliant
    v_sq   = cp.Variable(max_n, name='v_sq', nonneg=True)
    p_flow = cp.Variable(max_e, name='p_flow')
    q_flow = cp.Variable(max_e, name='q_flow')
    I_sq   = cp.Variable(max_e, name='I_sq', nonneg=True)
    y_line = cp.Variable(max_e, name='y_line')  # Relaxed to continuous [0,1]
    z_bus  = cp.Variable(max_n, name='z_bus')   # Relaxed to continuous [0,1]
    
    # Parameters (DPP requires these to be explicitly parameters)
    y_warm = cp.Parameter(max_e, name='y_warm')
    v_warm = cp.Parameter(max_n, name='v_warm')
    p_inj  = cp.Parameter(max_n, name='p_inj')
    q_inj  = cp.Parameter(max_n, name='q_inj')
    y0     = cp.Parameter(max_e, name='y0')

    # --- new: cable & topology parameters ---
    r_pu       = cp.Parameter(max_e, name='r_pu')           # line resistance
    x_pu       = cp.Parameter(max_e, name='x_pu')           # line reactance
    bigM_flow  = cp.Parameter(max_e, name='bigM_flow')      # M for flows
    A_from = cp.Parameter((max_e, max_n), name='A_from')# incidence-from
    A_to   = cp.Parameter((max_e, max_n), name='A_to')  # incidence-to
    bigM_v_param = cp.Parameter(nonneg=True, name='bigM_v')
    
    lb = cp.Parameter(name="zero", shape=(), value=0.0)
    ub = cp.Parameter(name="one",  shape=(), value=1.0)
    Vref = cp.Parameter(name="V_ref", shape=(), value=1.0)
    
    # --- 4. Build DPP-Compliant Constraints ---
    constraints = []
    
    # DPP FIX 2: Explicit bounds for relaxed binary variables
    constraints += [0 <= y_line, y_line <= 1]
    constraints += [0 <= z_bus, z_bus <= 1]
    
    # DPP FIX 3: Substation constraints using element-wise operations
    # Instead of cp.multiply, use element-wise constraints for substations
    for i in range(max_n):
        if sub_idx_np[i] > 0.5:  # This bus is a substation
            constraints += [v_sq[i] == 1.0]  # Fix voltage
            constraints += [z_bus[i] == 1.0]  # Fix energization
    
    # DPP FIX 4: Voltage bounds for non-substation buses
    # Use simple linear constraints instead of complex multiply operations
    for i in range(max_n):
        if non_sub_mask_np[i] > 0.5:  # This is a non-substation bus
            constraints += [ v_sq[i] >= cp.multiply(v_lower**2,           z_bus[i]) ]
            constraints += [ v_sq[i] <= v_upper**2 + cp.multiply(bigM_v_param, (1 - z_bus[i])) ]
    
    # DPP FIX 5: Line-bus linking constraints (vectorized but DPP-compliant)
    constraints += [y_line <= A_from @ z_bus]
    constraints += [y_line <= A_to @ z_bus]
    
    # DPP FIX 6: Power flow bounds using simple linear constraints
    constraints += [p_flow <= bigM_flow * y_line]
    constraints += [p_flow >= -bigM_flow * y_line]
    constraints += [q_flow <= bigM_flow * y_line]
    constraints += [q_flow >= -bigM_flow * y_line]
    #constraints += [I_sq    <= (bigM_flow**2) * y_line]
    constraints += [ I_sq <= cp.multiply(cp.square(bigM_flow), y_line) ]

    # DPP FIX 7: SOCP constraint using proper SOC formulation
    # Replace quadratic with second-order cone constraint
    # ||[P, Q, (V_i - I)/2]||_2 <= (V_i + I)/2 + bigM*(1-y)
    v_from = A_from @ v_sq
    
    # Create the SOC constraint matrix properly for DPP
    soc_lhs = cp.hstack([
        cp.reshape(p_flow, (max_e, 1)),
        cp.reshape(q_flow, (max_e, 1)), 
        cp.reshape((v_from - I_sq)/2, (max_e, 1))
    ])
    soc_rhs = (v_from + I_sq)/2 + cp.multiply(cp.square(bigM_flow), (1 - y_line))
    
    # DPP-compliant SOC constraint
    constraints += [cp.norm(soc_lhs, p=2, axis=1) <= soc_rhs]
    
    # DPP FIX 8: Voltage drop constraints using linear formulation
    # |voltage_drop| <= bigM * (1 - y_line) becomes two linear constraints
    #z_squared_term = cp.multiply(r_pu_np**2 + x_pu_np**2, I_sq)
    z_squared_term = cp.multiply(cp.square(r_pu) + cp.square(x_pu), I_sq)
    voltage_drop_vec = (A_to @ v_sq - A_from @ v_sq
                    + 2 * (cp.multiply(r_pu, p_flow)
                           + cp.multiply(x_pu, q_flow))
                    - z_squared_term)
    
    constraints += [voltage_drop_vec <= bigM_v * (1 - y_line)]
    constraints += [voltage_drop_vec >= -bigM_v * (1 - y_line)]

    # DPP FIX 9: Power balance using element-wise operations for clarity
    B = A_to - A_from  # both are cp.Parameter
    p_balance = B.T @ p_flow
    q_balance = B.T @ q_flow
    
    for i in range(max_n):
        if non_sub_mask_np[i] > 0.5:
            constraints += [ (B.T @ q_flow)[i] == q_inj[i] * z_bus[i] ]
            constraints += [ (B.T @ p_flow)[i] == p_inj[i] * z_bus[i] ]

    
    # DPP FIX 10: Spanning tree constraint (simple linear constraint)
    constraints += [cp.sum(y_line) == cp.sum(z_bus) - 1]
    
    # DPP FIX 11: Ensure connectivity (at least one substation energized)
    if np.sum(sub_idx_np) > 0:
        substation_energization = cp.sum([z_bus[i] for i in range(max_n) if sub_idx_np[i] > 0.5])
        constraints += [substation_energization >= 1]

    # --- 5. Define Objective (DPP-compliant) ---
    # DPP FIX 12: Use simple quadratic forms that are DPP-compliant
    #loss_objective = cp.sum(cp.multiply(r_pu_np, I_sq))
    loss_objective = cp.sum(cp.multiply(r_pu, I_sq))
    
    # Warm start penalties using sum_squares (DPP-compliant)
    loss_objective += 0.001 * cp.sum_squares(y_line - y_warm)
    loss_objective += 0.001 * cp.sum_squares(v_sq - v_warm)
    
    # Switch change penalty using L1 norm (DPP-compliant)
    loss_objective += 0.0001 * cp.norm(y_line - y0, 1)
    
    objective = cp.Minimize(loss_objective)

    problem = cp.Problem(objective, constraints)
    
    # Verify DPP compliance
    is_dcp = problem.is_dcp()
    is_dpp = problem.is_dcp(dpp=True)
    logger.info(f"CVXPY problem is DCP: {is_dcp}")
    logger.info(f"CVXPY problem is DPP: {is_dpp}")
    
    if not is_dpp:
        logger.error("Problem is not DPP compliant! Check constraint formulations.")
        # Print constraint details for debugging
        for i, constraint in enumerate(constraints[:5]):  # Print first few constraints
            logger.error(f"Constraint {i}: {constraint}")
    
    cvx_layer = CvxpyLayer(problem,
                           parameters=[
                             y_warm, v_warm, p_inj, q_inj, y0,
                             r_pu, x_pu, bigM_flow, A_from, A_to,
                             bigM_v_param
                           ],)
    return cvx_layer

class cvx(nn.Module):
    def __init__(self, **kwargs):
        super(cvx, self).__init__()
        self.cvx_layer = kwargs['cvx_layer']
        hidden_dims = kwargs['hidden_dims']
        latent_dim = kwargs['latent_dim']

        self.node_encoder = nn.Linear(kwargs['node_input_dim'], hidden_dims[0])
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)
        ])
        self.gnn_layers.append(GCNConv(hidden_dims[-1], latent_dim))

        self.switch_predictor = nn.Sequential(nn.Linear(latent_dim * 2, 1), nn.Sigmoid())
        self.voltage_predictor = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=kwargs['dropout_rate'])
        self.v_lower = 0.9
        self.v_upper = 1.1
#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

#         # GNN forward pass
#         x = self.relu(self.node_encoder(x))
#         for conv in self.gnn_layers:
#             x = self.relu(conv(x, edge_index))
        
#         edge_emb = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        
#         switch_warm_start = self.switch_predictor(edge_emb).squeeze(-1)
#         voltage_warm_start = self.voltage_predictor(x).squeeze(-1)
#         v_pred_scaled = self.v_lower + (self.v_upper - self.v_lower) * voltage_warm_start
#         v_sq_warm_start = v_pred_scaled.pow(2)

#         # DIAGNOSTIC 1: Check warm start diversity
#         logger.info(f"WARM START ANALYSIS:")
#         logger.info(f"  Switch warm start - min: {switch_warm_start.min():.4f}, max: {switch_warm_start.max():.4f}")
#         logger.info(f"  Switch warm start - std: {switch_warm_start.std():.4f}, mean: {switch_warm_start.mean():.4f}")
#         logger.info(f"  Switch warm start - unique values: {len(torch.unique(switch_warm_start))}")
        
#         # Count how many warm starts are above/below 0.5
#         warm_above_half = (switch_warm_start > 0.5).sum().item()
#         warm_below_half = (switch_warm_start <= 0.5).sum().item()
#         logger.info(f"  Warm starts > 0.5: {warm_above_half}, <= 0.5: {warm_below_half}")

#         # Get CVX parameters
#         p_inj_t = data.cvx_p_inj.squeeze(-1)
#         q_inj_t = data.cvx_q_inj.squeeze(-1)
#         y0_t = data.cvx_y0.squeeze(-1)

#         # DIAGNOSTIC 2: Check input parameter diversity
#         logger.info(f"CVX PARAMETER ANALYSIS:")
#         logger.info(f"  p_inj range: [{p_inj_t.min():.4f}, {p_inj_t.max():.4f}], std: {p_inj_t.std():.4f}")
#         logger.info(f"  q_inj range: [{q_inj_t.min():.4f}, {q_inj_t.max():.4f}], std: {q_inj_t.std():.4f}")
#         logger.info(f"  y0 range: [{y0_t.min():.4f}, {y0_t.max():.4f}], std: {y0_t.std():.4f}")

#         # Ensure warm starts are in valid range
#         switch_warm_start = torch.clamp(switch_warm_start, 0.01, 0.99)
#         v_sq_warm_start = torch.clamp(v_sq_warm_start, (self.v_lower**2)*0.95, (self.v_upper**2)*1.05)

#         try:
#             # DIAGNOSTIC 3: Test CVX layer with different inputs
#             # Test 1: Use current warm starts
#             y_opt1, v_sq_opt1 = self.cvx_layer(
#                 switch_warm_start, v_sq_warm_start,
#                 p_inj_t, q_inj_t, y0_t,
#                 solver_args={'verbose': False, 'max_iters': 5000, 'eps': 1e-5}
#             )
            
#             # Test 2: Use uniform warm starts to see if CVX is sensitive
#             uniform_switches = torch.full_like(switch_warm_start, 0.5)
#             uniform_voltages = torch.full_like(v_sq_warm_start, 1.0)
            
#             y_opt2, v_sq_opt2 = self.cvx_layer(
#                 uniform_switches, uniform_voltages,
#                 p_inj_t, q_inj_t, y0_t,
#                 solver_args={'verbose': False, 'max_iters': 5000, 'eps': 1e-5}
#             )
            
#             # Test 3: Use random warm starts
#             random_switches = torch.rand_like(switch_warm_start) * 0.8 + 0.1  # [0.1, 0.9]
#             random_voltages = torch.rand_like(v_sq_warm_start) * 0.4 + 0.8    # [0.8, 1.2]
            
#             y_opt3, v_sq_opt3 = self.cvx_layer(
#                 random_switches, random_voltages,
#                 p_inj_t, q_inj_t, y0_t,
#                 solver_args={'verbose': False, 'max_iters': 5000, 'eps': 1e-5}
#             )
            
#             # DIAGNOSTIC 4: Compare outputs
#             logger.info(f"CVX OUTPUT COMPARISON:")
#             logger.info(f"  Test 1 (GNN warm start) - switches std: {y_opt1.std():.6f}, unique: {len(torch.unique(y_opt1.round(decimals=4)))}")
#             logger.info(f"  Test 2 (uniform warm start) - switches std: {y_opt2.std():.6f}, unique: {len(torch.unique(y_opt2.round(decimals=4)))}")
#             logger.info(f"  Test 3 (random warm start) - switches std: {y_opt3.std():.6f}, unique: {len(torch.unique(y_opt3.round(decimals=4)))}")
            
#             # Check if outputs are nearly identical (indicates over-constrained problem)
#             diff_1_2 = torch.abs(y_opt1 - y_opt2).max().item()
#             diff_1_3 = torch.abs(y_opt1 - y_opt3).max().item()
#             logger.info(f"  Max difference between Test 1 & 2: {diff_1_2:.6f}")
#             logger.info(f"  Max difference between Test 1 & 3: {diff_1_3:.6f}")
            
#             if diff_1_2 < 1e-4 and diff_1_3 < 1e-4:
#                 logger.error("WARNING: CVX outputs are nearly identical regardless of warm start!")
#                 logger.error("This suggests the problem is over-constrained or has a unique solution.")
            
#             # Use the original warm start result
#             y_opt, v_sq_opt = y_opt1, v_sq_opt1
            
#         except Exception as e:
#             logger.error(f"CVX layer failed: {e}. Using warm-starts as fallback.")
#             y_opt = switch_warm_start
#             v_sq_opt = v_sq_warm_start

#         # DIAGNOSTIC 5: Final output analysis
#         logger.info(f"FINAL OUTPUT ANALYSIS:")
#         logger.info(f"  y_opt range: [{y_opt.min():.4f}, {y_opt.max():.4f}]")
#         logger.info(f"  y_opt std: {y_opt.std():.6f}")
#         logger.info(f"  y_opt unique values: {len(torch.unique(y_opt.round(decimals=4)))}")
        
#         # Count predictions above/below threshold
#         pred_above_half = (y_opt > 0.5).sum().item()
#         pred_below_half = (y_opt <= 0.5).sum().item()
#         logger.info(f"  Predictions > 0.5: {pred_above_half}, <= 0.5: {pred_below_half}")
        
#         # Check if warm starts were ignored
#         warm_start_ignored = torch.abs(y_opt - switch_warm_start).mean().item()
#         logger.info(f"  Mean change from warm start: {warm_start_ignored:.6f}")

#         return {
#             "switch_predictions": switch_warm_start,
#             "voltage_predictions": v_sq_warm_start,
#             "switch_scores": y_opt,
#             "voltage_optimal": v_sq_opt
#         }
# def diagnose_cvx_layer_constraints(cvx_layer, sample_data):
#     """Test the CVX layer with various inputs to understand constraint behavior."""
    
#     max_e = sample_data.edge_index.shape[1]
#     max_n = sample_data.num_nodes
    
#     # Test cases
#     test_cases = [
#         ("all_on", torch.ones(max_e) * 0.9, torch.ones(max_n)),
#         ("all_off", torch.ones(max_e) * 0.1, torch.ones(max_n)),
#         ("half_half", torch.cat([torch.ones(max_e//2) * 0.9, torch.ones(max_e - max_e//2) * 0.1]), torch.ones(max_n)),
#         ("random", torch.rand(max_e), torch.rand(max_n) * 0.4 + 0.8),
#     ]
    
#     p_inj = sample_data.cvx_p_inj.squeeze(-1)
#     q_inj = sample_data.cvx_q_inj.squeeze(-1)
#     y0 = sample_data.cvx_y0.squeeze(-1)
    
#     logger.info("=== CVX LAYER CONSTRAINT DIAGNOSIS ===")
    
#     for name, y_warm, v_warm in test_cases:
#         try:
#             y_opt, v_opt = cvx_layer(y_warm, v_warm, p_inj, q_inj, y0)
            
#             # Check spanning tree constraint
#             tree_constraint = y_opt.sum() - (max_n - 1)  # Should be ~0 for spanning tree
            
#             logger.info(f"Test '{name}':")
#             logger.info(f"  Input y range: [{y_warm.min():.3f}, {y_warm.max():.3f}]")
#             logger.info(f"  Output y range: [{y_opt.min():.3f}, {y_opt.max():.3f}]")
#             logger.info(f"  Tree constraint violation: {tree_constraint:.6f}")
#             logger.info(f"  Output std: {y_opt.std():.6f}")
            
#         except Exception as e:
#             logger.error(f"Test '{name}' failed: {e}")
    
#     logger.info("=== END DIAGNOSIS ===")


    def forward(self, data):
        data = data.to(self.device)

        # true sizes
        N = data.cvx_N[0].item()
        E = data.cvx_E[0].item()

        # retrieve padded parameters
        p_inj     = data.cvx_p_inj[0]      # (max_N,)
        q_inj     = data.cvx_q_inj[0]      # (max_N,)
        y0        = data.cvx_y0[0]         # (max_E,)
        r_pu      = data.cvx_r_pu[0]       # (max_E,)
        x_pu      = data.cvx_x_pu[0]       # (max_E,)
        bigM_flow = data.cvx_bigM_flow[0]  # (max_E,)

        from_idx  = data.cvx_from_idx[0].long()   # (max_E,)
        to_idx    = data.cvx_to_idx  [0].long()   # (max_E,)
        node_mask = data.cvx_node_mask[0].bool()  # (max_N,)
        edge_mask = data.cvx_edge_mask[0].bool()  # (max_E,)

        # build incidence matrices on the fly
        A_from = torch.zeros((self.max_e, self.max_n), device=self.device)
        A_to   = torch.zeros_like(A_from)
        idx = torch.arange(E, device=self.device)
        A_from[idx, from_idx[idx]] = 1.0
        A_to  [idx,   to_idx  [idx]]   = 1.0

        # GNN warm-start prediction
        x = self.relu(self.node_encoder(data.x))
        for conv in self.gnn_layers:
            x = self.relu(conv(x, data.edge_index))
        edge_emb = torch.cat([x[data.edge_index[0]], x[data.edge_index[1]]], dim=1)
        y_warm = self.switch_predictor(edge_emb).squeeze(-1)       # (max_E,)
        v_raw  = self.voltage_predictor(x).squeeze(-1)            # (max_N,)
        v_warm = (self.v_lower + (self.v_upper - self.v_lower) * v_raw).pow(2)

        # DIAGNOSTIC: warm start
        self.logger.info("WARM START ANALYSIS:")
        self.logger.info("  y_warm min/max: %f / %f", y_warm.min().item(), y_warm.max().item())
        self.logger.info("  y_warm std/mean: %f / %f", y_warm.std().item(), y_warm.mean().item())
        self.logger.info("  unique y_warm: %d", y_warm.unique().numel())
        self.logger.info("  y_warm >0.5: %d, <=0.5: %d",
                        (y_warm>0.5).sum().item(), (y_warm<=0.5).sum().item())

        # DIAGNOSTIC: CVX input params
        self.logger.info("CVX PARAMETER ANALYSIS:")
        self.logger.info("  p_inj range/std: [%f,%f] / %f",
                        p_inj[:N].min().item(), p_inj[:N].max().item(), p_inj[:N].std().item())
        self.logger.info("  q_inj range/std: [%f,%f] / %f",
                        q_inj[:N].min().item(), q_inj[:N].max().item(), q_inj[:N].std().item())
        self.logger.info("  y0   range/std: [%f,%f] / %f",
                        y0[:E].min().item(),    y0[:E].max().item(),    y0[:E].std().item())

        # Solve CVX layer
        y_opt, v_sq_opt = self.cvx_layer(
            y_warm, v_warm,
            p_inj, q_inj, y0,
            r_pu, x_pu, bigM_flow,
            A_from, A_to,
            solver_args={'verbose': False}
        )

        # DIAGNOSTIC: CVX outputs
        self.logger.info("FINAL OUTPUT ANALYSIS:")
        self.logger.info("  y_opt range: [%f,%f]",       y_opt.min().item(), y_opt.max().item())
        self.logger.info("  y_opt std: %f",              y_opt.std().item())
        self.logger.info("  unique y_opt: %d",           y_opt.unique().numel())
        self.logger.info("  y_opt >0.5: %d, <=0.5: %d",
                        (y_opt>0.5).sum().item(), (y_opt<=0.5).sum().item())
        self.logger.info("  mean |y_opt - y_warm|: %f",
                        (y_opt - y_warm).abs().mean().item())

        # strip padded entries for downstream metrics
        # y_opt    = y_opt[edge_mask]
        # v_sq_opt = v_sq_opt[node_mask]

        return y_opt, v_sq_opt
