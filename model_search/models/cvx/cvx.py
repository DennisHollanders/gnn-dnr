import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

def build_cvx_layer(max_n: int, max_e: int):
    """Build DPP-compliant CVX layer using affine constraints only"""
    # Full-size variables (using padded dimensions)
    v_sq         = cp.Variable(max_n, nonneg=True)
    p_flow       = cp.Variable(max_e)
    q_flow       = cp.Variable(max_e)
    I_sq         = cp.Variable(max_e, nonneg=True)
    y_line       = cp.Variable(max_e, nonneg=True)
    z_bus        = cp.Variable(max_n, nonneg=True)

    # Warm start parameters (padded size)
    y_warm       = cp.Parameter(max_e, nonneg=True)
    v_warm       = cp.Parameter(max_n, nonneg=True)
    
    # CVX parameters (already padded in your data pipeline)
    p_inj_full   = cp.Parameter(max_n)
    q_inj_full   = cp.Parameter(max_n)
    y0_full      = cp.Parameter(max_e, nonneg=True)
    r_pu_full    = cp.Parameter(max_e, nonneg=True)
    x_pu_full    = cp.Parameter(max_e, nonneg=True)
    bigM_flow_full = cp.Parameter(max_e, nonneg=True)
    bigM_v       = cp.Parameter(nonneg=True)
    A_from_full  = cp.Parameter((max_e, max_n), nonneg=True)
    A_to_full    = cp.Parameter((max_e, max_n), nonneg=True)
    sub_mask_full = cp.Parameter(max_n, nonneg=True)
    non_sub_mask_full = cp.Parameter(max_n, nonneg=True)
    bigM_flow_sq_full = cp.Parameter(max_e, nonneg=True)
    z_line_sq_full = cp.Parameter(max_e, nonneg=True)
    
    # DPP-compliant masking using selection matrices (diagonal matrices)
    S_nodes      = cp.Parameter((max_n, max_n), nonneg=True)  # Diagonal selection matrix for nodes
    S_edges      = cp.Parameter((max_e, max_e), nonneg=True)  # Diagonal selection matrix for edges
    
    # Target values for inactive elements (DPP-compliant)
    v_target     = cp.Parameter(max_n, nonneg=True)  # Target voltages (1.0 for inactive, 0.0 for active)
    z_target     = cp.Parameter(max_n, nonneg=True)  # Target z_bus (0.0 for all)
    y_target     = cp.Parameter(max_e, nonneg=True)  # Target y_line (0.0 for all)
    flow_target  = cp.Parameter(max_e)               # Target flows (0.0 for all)
    I_target     = cp.Parameter(max_e, nonneg=True)  # Target currents (0.0 for all)

    v_low_sq  = 0.9**2
    v_high_sq = 1.1**2

    cons = []
    
    # Basic bounds
    cons += [y_line >= 0, y_line <= 1]
    cons += [z_bus >= 0, z_bus <= 1]
    cons += [v_sq >= v_low_sq*0.8, v_sq <= v_high_sq*1.2]
    
    # DPP-compliant masking using affine constraints
    # For active elements: S @ x = x, target = 0 → constraint becomes x = x (no effect)
    # For inactive elements: S @ x = 0, target = value → constraint becomes x = value
    cons += [v_sq == S_nodes @ v_sq + v_target]     # Active: no change, Inactive: v_sq = 1.0
    cons += [z_bus == S_nodes @ z_bus + z_target]   # Active: no change, Inactive: z_bus = 0.0
    cons += [y_line == S_edges @ y_line + y_target] # Active: no change, Inactive: y_line = 0.0
    cons += [p_flow == S_edges @ p_flow + flow_target] # Active: no change, Inactive: p_flow = 0.0
    cons += [q_flow == S_edges @ q_flow + flow_target] # Active: no change, Inactive: q_flow = 0.0
    cons += [I_sq == S_edges @ I_sq + I_target]     # Active: no change, Inactive: I_sq = 0.0
    
    # Substation constraints (applied to all, but inactive nodes are fixed)
    cons += [cp.multiply(sub_mask_full, v_sq) == sub_mask_full]
    cons += [cp.multiply(sub_mask_full, z_bus) == sub_mask_full]

    # Voltage bounds for non-substation nodes
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
    
    # Flow constraints
    cons += [
        p_flow <= cp.multiply(bigM_flow_full, y_line),
       -p_flow <= cp.multiply(bigM_flow_full, y_line),
        q_flow <= cp.multiply(bigM_flow_full, y_line),
       -q_flow <= cp.multiply(bigM_flow_full, y_line),
        I_sq   <= cp.multiply(bigM_flow_sq_full, y_line)
    ]

    # SOC constraint (applied to all edges, inactive edges have 0 flows due to masking)
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

    # Radiality constraint - uses sum over all elements, but inactive are 0 due to masking
    cons += [cp.sum(y_line) == cp.sum(z_bus) - 1]

    # Objective function
    loss  = cp.sum(cp.multiply(r_pu_full, I_sq))
    loss += 0.001*cp.sum_squares(y_line - y_warm)
    loss += 0.001*cp.sum_squares(v_sq - v_warm)
    loss += 0.0001*cp.norm1(y_line - y0_full)

    problem = cp.Problem(cp.Minimize(loss), cons)
    
    # Verify DPP compliance
    if not problem.is_dcp(dpp=True):
        print("Warning: Problem is not DPP compliant!")
        # Try to identify which constraints are problematic
        for i, constraint in enumerate(cons):
            if not constraint.is_dcp(dpp=True):
                print(f"Constraint {i} is not DPP compliant: {constraint}")

    return CvxpyLayer(
        problem,
        parameters=[
            y_warm, v_warm, p_inj_full, q_inj_full, y0_full, r_pu_full, x_pu_full,
            bigM_flow_full, bigM_v, A_from_full, A_to_full, sub_mask_full, non_sub_mask_full,
            bigM_flow_sq_full, z_line_sq_full, S_nodes, S_edges,
            v_target, z_target, y_target, flow_target, I_target
        ],
        variables=[y_line, v_sq]
    )

class cvx(nn.Module):
    def __init__(self, **K):
        super().__init__()
        self.max_n = K['max_n']
        self.max_e = K['max_e']
        self.cvx_layer = build_cvx_layer(self.max_n, self.max_e)
        
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
        """Create DPP-compliant selection matrices and target vectors"""
        # Selection matrices: diagonal with 1s for active elements, 0s for inactive
        S_nodes = torch.zeros(self.max_n, self.max_n, device=device)
        S_edges = torch.zeros(self.max_e, self.max_e, device=device)
        
        # Set diagonal elements to 1 for active nodes/edges
        S_nodes[:actual_nodes, :actual_nodes] = torch.eye(actual_nodes, device=device)
        S_edges[:actual_edges, :actual_edges] = torch.eye(actual_edges, device=device)
        
        # Target values: 0 for active elements, desired value for inactive elements
        v_target = torch.ones(self.max_n, device=device)  # Inactive nodes → v_sq = 1.0
        v_target[:actual_nodes] = 0.0  # Active nodes → add 0 (no change)
        
        z_target = torch.zeros(self.max_n, device=device)  # All → z_bus gets 0 added
        y_target = torch.zeros(self.max_e, device=device)  # All → y_line gets 0 added
        flow_target = torch.zeros(self.max_e, device=device)  # All → flows get 0 added
        I_target = torch.zeros(self.max_e, device=device)  # All → currents get 0 added
        
        return S_nodes, S_edges, v_target, z_target, y_target, flow_target, I_target

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = self.relu(self.node_enc(x))
        for g in self.gnns:
            x = self.relu(g(x, ei))
        
        # GNN predictions (unpadded - based on actual graph size)
        emb = torch.cat([x[ei[0]], x[ei[1]]], dim=1)
        yw_unpadded = self.sw_pred(emb).squeeze(-1)  # Shape: [actual_edges]
        vr = self.v_pred(x).squeeze(-1)              # Shape: [actual_nodes]
        vw_unpadded = (self.vl + (self.vu - self.vl)*vr).pow(2)
        
        # Get actual sizes from GNN
        actual_edges = ei.shape[1]
        actual_nodes = x.shape[0]
        
        #print(f"Debug: GNN output shapes - nodes: {actual_nodes}, edges: {actual_edges}")
        
        # Create warm start by padding GNN predictions to CVX size
        yw = torch.zeros(self.max_e, device=yw_unpadded.device, dtype=yw_unpadded.dtype)
        yw[:actual_edges] = torch.clamp(yw_unpadded, min=0.0, max=1.0)
        
        vw = torch.ones(self.max_n, device=vw_unpadded.device, dtype=vw_unpadded.dtype)
        vw[:actual_nodes] = torch.clamp(vw_unpadded, min=0.81, max=1.21)
        
        # Create DPP-compliant selection matrices and targets
        S_nodes, S_edges, v_target, z_target, y_target, flow_target, I_target = \
            self.create_selection_matrices_and_targets(actual_nodes, actual_edges, x.device)
        
        try:
            # Call CVX layer with DPP-compliant parameters
            y_opt, v_sq_opt = self.cvx_layer(
                yw, vw,
                data.cvx_p_inj.squeeze(0), data.cvx_q_inj.squeeze(0), data.cvx_y0.squeeze(0),
                data.cvx_r_pu.squeeze(0), data.cvx_x_pu.squeeze(0), 
                data.cvx_bigM_flow.squeeze(0), data.cvx_bigM_v.squeeze(0),
                data.cvx_A_from.squeeze(0), data.cvx_A_to.squeeze(0), 
                data.cvx_sub_mask.squeeze(0), data.cvx_non_sub_mask.squeeze(0),
                data.cvx_bigM_flow_sq.squeeze(0), data.cvx_z_line_sq.squeeze(0),
                S_nodes, S_edges, v_target, z_target, y_target, flow_target, I_target,
                solver_args={'verbose': False, 'solve_method': 'ECOS'}
            )
            
            print(f"Debug: CVX output shapes - nodes: {v_sq_opt}, edges: {y_opt}")
            # Extract only the relevant parts (first actual_edges/actual_nodes)
            y_opt_unpadded = y_opt[:actual_edges]
            v_sq_opt_unpadded = v_sq_opt[:actual_nodes]

            print(f"Debug: CVX unpadded shapes - nodes: {v_sq_opt_unpadded}, edges: {y_opt_unpadded}")
            
            return {
                "switch_scores": y_opt_unpadded,
                "voltage_scores": v_sq_opt_unpadded,
                "switch_predictions": yw_unpadded,
                "voltage_predictions": vw_unpadded,
            }
            
        except Exception as e:
            print(f"CVX solver error: {e}")
            print("Falling back to GNN predictions only")
            return {
                "switch_scores": yw_unpadded,
                "voltage_scores": vw_unpadded.sqrt(),  # Convert back from squared
                "switch_predictions": yw_unpadded,
                "voltage_predictions": vw_unpadded,
            }