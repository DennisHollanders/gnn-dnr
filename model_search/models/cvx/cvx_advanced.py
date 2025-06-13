import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

def build_cvx_layer(max_n: int, max_e: int, tightening_method: str = "ccp"):
    """Build CVX layer with binary tightening methods"""
    
    # Variables
    v_sq = cp.Variable(max_n, nonneg=True)
    p_flow = cp.Variable(max_e)
    q_flow = cp.Variable(max_e)
    I_sq = cp.Variable(max_e, nonneg=True)
    y_line = cp.Variable(max_e, nonneg=True)
    z_bus = cp.Variable(max_n, nonneg=True)

    # Warm start and CCP parameters
    y_warm = cp.Parameter(max_e, nonneg=True)
    v_warm = cp.Parameter(max_n, nonneg=True)
    
    # CCP iteration parameter (for convex-concave procedure)
    y_prev = cp.Parameter(max_e, nonneg=True)  # Previous iteration values
    z_prev = cp.Parameter(max_n, nonneg=True)
    
    # Penalty weights
    penalty_weight = cp.Parameter(nonneg=True)  # Adaptive penalty weight
    
    # All other CVX parameters (same as before)
    p_inj_full = cp.Parameter(max_n)
    q_inj_full = cp.Parameter(max_n)
    y0_full = cp.Parameter(max_e, nonneg=True)
    r_pu_full = cp.Parameter(max_e, nonneg=True)
    x_pu_full = cp.Parameter(max_e, nonneg=True)
    bigM_flow_full = cp.Parameter(max_e, nonneg=True)
    bigM_v = cp.Parameter(nonneg=True)
    A_from_full = cp.Parameter((max_e, max_n), nonneg=True)
    A_to_full = cp.Parameter((max_e, max_n), nonneg=True)
    sub_mask_full = cp.Parameter(max_n, nonneg=True)
    non_sub_mask_full = cp.Parameter(max_n, nonneg=True)
    bigM_flow_sq_full = cp.Parameter(max_e, nonneg=True)
    z_line_sq_full = cp.Parameter(max_e, nonneg=True)
    
    # DPP-compliant masking
    S_nodes = cp.Parameter((max_n, max_n), nonneg=True)
    S_edges = cp.Parameter((max_e, max_e), nonneg=True)
    v_target = cp.Parameter(max_n, nonneg=True)
    z_target = cp.Parameter(max_n, nonneg=True)
    y_target = cp.Parameter(max_e, nonneg=True)
    flow_target = cp.Parameter(max_e)
    I_target = cp.Parameter(max_e, nonneg=True)

    v_low_sq = 0.9**2
    v_high_sq = 1.1**2

    cons = []
    
    # Basic bounds with tighter constraints near binary values
    cons += [y_line >= 0, y_line <= 1]
    cons += [z_bus >= 0, z_bus <= 1]
    cons += [v_sq >= v_low_sq*0.8, v_sq <= v_high_sq*1.2]
    
    # DPP-compliant masking (same as before)
    cons += [v_sq == S_nodes @ v_sq + v_target]
    cons += [z_bus == S_nodes @ z_bus + z_target]
    cons += [y_line == S_edges @ y_line + y_target]
    cons += [p_flow == S_edges @ p_flow + flow_target]
    cons += [q_flow == S_edges @ q_flow + flow_target]
    cons += [I_sq == S_edges @ I_sq + I_target]
    
    # All other constraints (same as your original)
    cons += [cp.multiply(sub_mask_full, v_sq) == sub_mask_full]
    cons += [cp.multiply(sub_mask_full, z_bus) == sub_mask_full]
    
    mask_z = cp.multiply(non_sub_mask_full, z_bus)
    cons += [cp.multiply(non_sub_mask_full, v_sq) >= cp.multiply(mask_z, v_low_sq)]
    cons += [
        cp.multiply(non_sub_mask_full, v_sq)
        <= cp.multiply(non_sub_mask_full, v_high_sq)
         + cp.multiply(bigM_v, 1 - z_bus)
    ]

    # Topology constraints
    cons += [y_line <= A_from_full @ z_bus]
    cons += [y_line <= A_to_full @ z_bus]
    cons += [y_line >= A_from_full @ z_bus + A_to_full @ z_bus - 1]
    
    # Flow constraints
    cons += [
        p_flow <= cp.multiply(bigM_flow_full, y_line),
       -p_flow <= cp.multiply(bigM_flow_full, y_line),
        q_flow <= cp.multiply(bigM_flow_full, y_line),
       -q_flow <= cp.multiply(bigM_flow_full, y_line),
        I_sq   <= cp.multiply(bigM_flow_sq_full, y_line)
    ]

    # SOC constraint
    v_from = A_from_full @ v_sq
    soc_X = cp.vstack([2 * p_flow, 2 * q_flow, v_from - I_sq])
    cons += [cp.SOC(v_from + I_sq, soc_X)]

    # Voltage drop equations
    vd = (A_to_full @ v_sq - v_from
         + 2*(cp.multiply(r_pu_full, p_flow) + cp.multiply(x_pu_full, q_flow))
         - cp.multiply(z_line_sq_full, I_sq))
    cons += [
        vd <= cp.multiply(bigM_v, 1 - y_line),
        vd >= cp.multiply(bigM_v, y_line - 1)
    ]

    # Power balance
    pb = (A_from_full - A_to_full).T @ p_flow
    qb = (A_from_full - A_to_full).T @ q_flow
    pe = pb - cp.multiply(p_inj_full, z_bus)
    qe = qb - cp.multiply(q_inj_full, z_bus)
    Mbal = 1e4
    cons += [
        pe <=  Mbal*sub_mask_full, -pe <=  Mbal*sub_mask_full,
        qe <=  Mbal*sub_mask_full, -qe <=  Mbal*sub_mask_full
    ]

    # Flow conservation (directed arc formulation)
    A_from_dir = cp.vstack([A_from_full, A_to_full])
    A_to_dir = cp.vstack([A_to_full, A_from_full])
    f = cp.Variable(2*max_e, nonneg=True)
    y_dup = cp.hstack([y_line, y_line])
    bigM2 = cp.hstack([bigM_flow_full, bigM_flow_full])
    cons += [f <= cp.multiply(bigM2, y_dup)]
    
    inflow = A_to_dir.T @ f
    outflow = A_from_dir.T @ f
    rhs = cp.hstack([cp.sum(z_bus) - 1, -z_bus[1:]])
    cons += [outflow - inflow == rhs]

    # Radiality constraint
    cons += [cp.sum(y_line) == cp.sum(z_bus) - 1]

    # BINARY TIGHTENING METHODS
    if tightening_method == "ccp":
        # Boyd's Convex-Concave Procedure
        # g(x) = x² - x is concave, so we linearize it at previous point
        # Linearization: g(x_prev) + ∇g(x_prev)^T(x - x_prev) = x_prev² - x_prev + (2*x_prev - 1)(x - x_prev)
        ccp_y = cp.sum(cp.multiply(2*y_prev - 1, y_line) - cp.multiply(y_prev, y_prev) + y_prev)
        ccp_z = cp.sum(cp.multiply(2*z_prev - 1, z_bus) - cp.multiply(z_prev, z_prev) + z_prev)
        binary_penalty = penalty_weight * (ccp_y + ccp_z)
        
    elif tightening_method == "quadratic":
        # Standard quadratic penalty: sum(x*(1-x))
        binary_penalty = penalty_weight * (cp.sum(cp.multiply(y_line, 1 - y_line)) + 
                                         cp.sum(cp.multiply(z_bus, 1 - z_bus)))
        
    elif tightening_method == "log_barrier":
        # Log barrier approximation: -ε*log(x) - ε*log(1-x)
        # Approximated using perspective function for DCP compliance
        eps = 0.01
        # Use quad_over_lin as a convex approximation to -log
        log_barrier_y = cp.sum(cp.quad_over_lin(cp.ones(max_e), y_line + eps) + 
                              cp.quad_over_lin(cp.ones(max_e), 1 - y_line + eps))
        log_barrier_z = cp.sum(cp.quad_over_lin(cp.ones(max_n), z_bus + eps) + 
                              cp.quad_over_lin(cp.ones(max_n), 1 - z_bus + eps))
        binary_penalty = penalty_weight * eps * (log_barrier_y + log_barrier_z)
        
    elif tightening_method == "adaptive_penalty":
        # Adaptive penalty that increases as values approach 0.5
        # Penalty = w * sum(4*x*(1-x)) - heaviest penalty at x=0.5
        adaptive_y = cp.sum(4 * cp.multiply(y_line, 1 - y_line))
        adaptive_z = cp.sum(4 * cp.multiply(z_bus, 1 - z_bus))
        binary_penalty = penalty_weight * (adaptive_y + adaptive_z)
        
    elif tightening_method == "combined":
        # Combination of methods
        ccp_y = cp.sum(cp.multiply(2*y_prev - 1, y_line) - cp.multiply(y_prev, y_prev) + y_prev)
        ccp_z = cp.sum(cp.multiply(2*z_prev - 1, z_bus) - cp.multiply(z_prev, z_prev) + z_prev)
        quad_penalty = cp.sum(cp.multiply(y_line, 1 - y_line)) + cp.sum(cp.multiply(z_bus, 1 - z_bus))
        binary_penalty = penalty_weight * (0.5 * (ccp_y + ccp_z) + 0.5 * quad_penalty)
        
    else:
        # Default: standard quadratic penalty
        binary_penalty = penalty_weight * (cp.sum(cp.multiply(y_line, 1 - y_line)) + 
                                         cp.sum(cp.multiply(z_bus, 1 - z_bus)))

    # Objective function
    loss = cp.sum(cp.multiply(r_pu_full, I_sq))
    loss += 0.001*cp.sum_squares(y_line - y_warm)
    loss += 0.001*cp.sum_squares(v_sq - v_warm)
    loss += 0.001*cp.norm1(y_line - y0_full)
    loss += binary_penalty  # Now with stronger, adaptive penalty

    problem = cp.Problem(cp.Minimize(loss), cons)
    
    if not problem.is_dcp(dpp=True):
        print("Warning: Problem is not DPP compliant!")

    # Updated parameter list
    parameters = [
        y_warm, v_warm, y_prev, z_prev, penalty_weight,
        p_inj_full, q_inj_full, y0_full, r_pu_full, x_pu_full,
        bigM_flow_full, bigM_v, A_from_full, A_to_full, sub_mask_full, non_sub_mask_full,
        bigM_flow_sq_full, z_line_sq_full, S_nodes, S_edges,
        v_target, z_target, y_target, flow_target, I_target
    ]

    return CvxpyLayer(problem, parameters=parameters, variables=[y_line, v_sq])


class cvx(nn.Module):
    def __init__(self, tightening_method="ccp", **K):
        super().__init__()
        self.max_n = K['max_n']
        self.max_e = K['max_e']
        self.tightening_method = tightening_method
        self.cvx_layer = build_cvx_layer(
            self.max_n, self.max_e, tightening_method
        )
        
        # CCP iteration tracking
        self.ccp_iterations = 0
        self.max_ccp_iterations = 5
        self.y_prev = None
        self.z_prev = None
        self.penalty_weight = 1.0  # Start with moderate penalty
        
        # GNN components (same as before)
        dims = K['hidden_dims']
        L = K['latent_dim']
        self.node_enc = nn.Linear(K['node_input_dim'], dims[0])
        self.gnns = nn.ModuleList([GCNConv(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.gnns.append(GCNConv(dims[-1], L))
        self.sw_pred = nn.Sequential(nn.Linear(2*L, 1), nn.Sigmoid())
        self.v_pred = nn.Sequential(nn.Linear(L, 1), nn.Sigmoid())
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=K['dropout_rate'])
        self.vl, self.vu = 0.9, 1.1

    def create_selection_matrices_and_targets(self, actual_nodes, actual_edges, device):
        """Same as before"""
        S_nodes = torch.zeros(self.max_n, self.max_n, device=device)
        S_edges = torch.zeros(self.max_e, self.max_e, device=device)
        
        S_nodes[:actual_nodes, :actual_nodes] = torch.eye(actual_nodes, device=device)
        S_edges[:actual_edges, :actual_edges] = torch.eye(actual_edges, device=device)
        
        v_target = torch.ones(self.max_n, device=device)
        v_target[:actual_nodes] = 0.0
        
        z_target = torch.zeros(self.max_n, device=device)
        y_target = torch.zeros(self.max_e, device=device)
        flow_target = torch.zeros(self.max_e, device=device)
        I_target = torch.zeros(self.max_e, device=device)
        
        return S_nodes, S_edges, v_target, z_target, y_target, flow_target, I_target

    def update_penalty_weight(self, y_values, z_values):
        """Adaptive penalty weight based on how binary the current solution is"""
        y_binary_score = torch.mean(torch.minimum(y_values, 1 - y_values))  # 0 = binary, 0.5 = worst
        z_binary_score = torch.mean(torch.minimum(z_values, 1 - z_values))
        avg_binary_score = (y_binary_score + z_binary_score) / 2
        
        # Increase penalty if values are not binary enough
        if avg_binary_score > 0.1:  # If average distance from binary > 0.1
            self.penalty_weight = min(self.penalty_weight * 2, 100.0)
        else:
            self.penalty_weight = max(self.penalty_weight * 0.8, 0.1)
        
        return self.penalty_weight

    def forward(self, data):
        x, ei = data.x, data.edge_index
        
        # GNN forward pass
        x = self.relu(self.node_enc(x))
        for g in self.gnns:
            x = self.relu(g(x, ei))
        
        emb = torch.cat([x[ei[0]], x[ei[1]]], dim=1)
        yw_unpadded = self.sw_pred(emb).squeeze(-1)
        vr = self.v_pred(x).squeeze(-1)
        vw_unpadded = (self.vl + (self.vu - self.vl)*vr).pow(2)
        
        actual_edges = ei.shape[1]
        actual_nodes = x.shape[0]
        
        # Pad to CVX size
        yw = torch.zeros(self.max_e, device=yw_unpadded.device, dtype=yw_unpadded.dtype)
        yw[:actual_edges] = torch.clamp(yw_unpadded, min=0.0, max=1.0)
        
        vw = torch.ones(self.max_n, device=vw_unpadded.device, dtype=vw_unpadded.dtype)
        vw[:actual_nodes] = torch.clamp(vw_unpadded, min=0.81, max=1.21)
        
        # Initialize CCP if first iteration
        if self.y_prev is None:
            self.y_prev = yw.clone()
            self.z_prev = torch.ones(self.max_n, device=yw.device) * 0.5  # Start at 0.5 for CCP
            self.z_prev[:actual_nodes] = 0.8  # Active nodes start closer to 1
        
        S_nodes, S_edges, v_target, z_target, y_target, flow_target, I_target = \
            self.create_selection_matrices_and_targets(actual_nodes, actual_edges, x.device)
        
        try:
            # Call CVX with tightening parameters
            y_opt, v_sq_opt = self.cvx_layer(
                yw, vw, self.y_prev, self.z_prev, 
                torch.tensor(self.penalty_weight, device=x.device),
                data.cvx_p_inj.squeeze(0), data.cvx_q_inj.squeeze(0), data.cvx_y0.squeeze(0),
                data.cvx_r_pu.squeeze(0), data.cvx_x_pu.squeeze(0), 
                data.cvx_bigM_flow.squeeze(0), data.cvx_bigM_v.squeeze(0),
                data.cvx_A_from.squeeze(0), data.cvx_A_to.squeeze(0), 
                data.cvx_sub_mask.squeeze(0), data.cvx_non_sub_mask.squeeze(0),
                data.cvx_bigM_flow_sq.squeeze(0), data.cvx_z_line_sq.squeeze(0),
                S_nodes, S_edges, v_target, z_target, y_target, flow_target, I_target,
                solver_args={'verbose': False, 'solve_method': 'ECOS'}
            )
            
            # Update CCP iteration
            if self.tightening_method in ["ccp", "combined"]:
                self.y_prev = y_opt.detach().clone()
                self.z_prev = torch.ones(self.max_n, device=y_opt.device)
                self.z_prev[:actual_nodes] = 0.8  # Update only active node estimates
            
            # Adaptive penalty update
            self.update_penalty_weight(y_opt[:actual_edges], self.z_prev[:actual_nodes])
            
            # Apply thresholding to push toward binary
            y_binary = torch.where(y_opt > 0.5, 1.0, 0.0)
            
            y_opt_unpadded = y_opt[:actual_edges]
            v_sq_opt_unpadded = v_sq_opt[:actual_nodes]
            
            print(f"Binary score: {torch.mean(torch.minimum(y_opt_unpadded, 1 - y_opt_unpadded)):.3f}, "
                  f"Penalty: {self.penalty_weight:.2f}")
            
            return {
                "switch_logits": y_opt_unpadded,
                "voltage_scores": v_sq_opt_unpadded,
                "switch_predictions": yw_unpadded,
                "voltage_predictions": vw_unpadded,
                "switch_binary": y_binary[:actual_edges],  # Hard binary decisions
            }
            
        except Exception as e:
            print(f"CVX solver error: {e}")
            return {
                "switch_scores": yw_unpadded,
                "voltage_scores": vw_unpadded.sqrt(),
                "switch_predictions": yw_unpadded,
                "voltage_predictions": vw_unpadded,
            }


# Usage example:
def create_enhanced_model(**kwargs):
    """Factory function to create model with different tightening methods"""
    method = kwargs.get('tightening_method', 'ccp')
    return cvx(tightening_method=method, **kwargs)

# Available methods:
# - "ccp": Boyd's Convex-Concave Procedure
# - "quadratic": Standard quadratic penalty
# - "log_barrier": Log barrier approximation  
# - "adaptive_penalty": Penalty that adapts based on distance from binary
# - "combined": Combination of CCP and quadratic penalty