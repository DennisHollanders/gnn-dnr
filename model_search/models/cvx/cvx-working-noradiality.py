import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import logging

logger = logging.getLogger(__name__)

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
    cons += [v_sq == S_nodes @ v_sq + v_target]
    cons += [z_bus == S_nodes @ z_bus + z_target]
    cons += [y_line == S_edges @ y_line + y_target]
    cons += [p_flow == S_edges @ p_flow + flow_target]
    cons += [q_flow == S_edges @ q_flow + flow_target]
    cons += [I_sq == S_edges @ I_sq + I_target]
    
    # Substation constraints
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

    # Radiality constraint
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
        self.cvx_layer = K["cvx_layer"] 
        
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
        
        # Add flag for gradient checking
        self.debug_gradients = True
        self.pre_cvx_hook  = lambda grad: self._hook_log("Pre-CVX", grad)
        self.post_cvx_hook = lambda grad: self._hook_log("Post-CVX", grad)

    def _hook_log(self, stage, grad):
        if grad is None:
            logger.info(f"{stage} hook: grad is None")
            return None
        norm = grad.norm().item()
        logger.info(f"{stage} hook grad-norm: {norm:.6f}")
        return grad


    def create_selection_matrices_and_targets(self, actual_nodes, actual_edges, device):
        """Create DPP-compliant selection matrices and target vectors"""
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

    # def forward(self, data):
    #     x, ei = data.x, data.edge_index
        
    #     # Enable gradient computation for intermediate values
    #     x.requires_grad_(True)
        
    #     x = self.relu(self.node_enc(x))
    #     for g in self.gnns:
    #         x = self.relu(g(x, ei))
        
    #     # GNN predictions
    #     emb = torch.cat([x[ei[0]], x[ei[1]]], dim=1)
    #     yw_unpadded = self.sw_pred(emb).squeeze(-1)
    #     vr = self.v_pred(x).squeeze(-1)
    #     vw_unpadded = (self.vl + (self.vu - self.vl)*vr).pow(2)
        
    #     # Get actual sizes
    #     actual_edges = ei.shape[1]
    #     actual_nodes = x.shape[0]

    #     pad_e = self.max_e - actual_edges
    #     if pad_e > 0:
    #         pad = yw_unpadded.new_zeros((pad_e,))
    #         yw = torch.cat([yw_unpadded, pad], dim=0)
    #     else:
    #         yw = yw_unpadded

    #     pad_n = self.max_n - actual_nodes
    #     if pad_n > 0:
    #         pad = vw_unpadded.new_ones((pad_n,))
    #         vw = torch.cat([vw_unpadded, pad], dim=0)
    #     else:
    #         vw = vw_unpadded

    #     self.last_yw, self.last_vw = yw, vw

    #     yw.register_hook(self.pre_cvx_hook)
    #     vw.register_hook(self.pre_cvx_hook)
        
        
    #     # Create DPP-compliant selection matrices
    #     S_nodes, S_edges, v_target, z_target, y_target, flow_target, I_target = \
    #         self.create_selection_matrices_and_targets(actual_nodes, actual_edges, x.device)
        
    #     try:
    #         # Call CVX layer
    #         y_opt, v_sq_opt = self.cvx_layer(
    #             yw, vw,
    #             data.cvx_p_inj.squeeze(0), data.cvx_q_inj.squeeze(0), data.cvx_y0.squeeze(0),
    #             data.cvx_r_pu.squeeze(0), data.cvx_x_pu.squeeze(0), 
    #             data.cvx_bigM_flow.squeeze(0), data.cvx_bigM_v.squeeze(0),
    #             data.cvx_A_from.squeeze(0), data.cvx_A_to.squeeze(0), 
    #             data.cvx_sub_mask.squeeze(0), data.cvx_non_sub_mask.squeeze(0),
    #             data.cvx_bigM_flow_sq.squeeze(0), data.cvx_z_line_sq.squeeze(0),
    #             S_nodes, S_edges, v_target, z_target, y_target, flow_target, I_target,
    #             solver_args={'verbose': False, 'solve_method': 'ECOS'}
    #         )
            
    #         # Extract relevant parts
    #         y_opt_unpadded = y_opt[:actual_edges]
    #         v_sq_opt_unpadded = v_sq_opt[:actual_nodes]

    #         y_opt_unpadded.register_hook(self.post_cvx_hook)
    #         v_sq_opt_unpadded.register_hook(self.post_cvx_hook)


    #         return {
    #             "switch_predictions": y_opt_unpadded,  # Changed from switch_logits
    #             "voltage_predictions": v_sq_opt_unpadded.sqrt(),  # Convert from squared
    #             "switch_logits": yw_unpadded,  # Keep GNN predictions as logits
    #             "voltage_scores": vw_unpadded,
    #         }
            
    #     except Exception as e:
    #         logger.error(f"CVX solver error: {e}")
    #         logger.info("Falling back to GNN predictions only")
    #         return {
    #             "switch_predictions": yw_unpadded,
    #             "voltage_predictions": vw_unpadded.sqrt(),
    #             "switch_logits": yw_unpadded,
    #             "voltage_scores": vw_unpadded,
    #         }

    def forward(self, data):
        # 1) GNN → raw logits
        x, ei = data.x, data.edge_index
        x = self.relu(self.node_enc(x))
        for g in self.gnns:
            x = self.relu(g(x, ei))

        emb = torch.cat([x[ei[0]], x[ei[1]]], dim=1)
        yw_unp = self.sw_pred(emb).squeeze(-1)      # (E,)
        vr     = self.v_pred(x).squeeze(-1)
        vw_unp = (self.vl + (self.vu - self.vl)*vr).pow(2)  # (N,)

        E, N = ei.shape[1], x.shape[0]

        # in __init__ or once per batch:
        P_e = torch.eye(self.max_e)[:E]   # shape [E, max_e]
        P_n = torch.eye(self.max_n)[:N]   # shape [N, max_n]

        # 2) Functional padding
        pad_e = self.max_e - E
        yw_pad = yw_unp if pad_e==0 else torch.cat([yw_unp, yw_unp.new_zeros(pad_e)], 0)

        pad_n = self.max_n - N
        vw_pad = vw_unp if pad_n==0 else torch.cat([vw_unp, vw_unp.new_ones(pad_n)], 0)

        # 3) Make them require_grad (so register_hook works)
        #    but keep the grad_fn so gradients still flow back through the GNN.
        yw = yw_pad.clone().requires_grad_()
        vw = vw_pad.clone().requires_grad_()

        # stash for logging after backward
        self.last_yw, self.last_vw = yw, vw

        # 4) register your hooks
        yw.register_hook(self.pre_cvx_hook)
        vw.register_hook(self.pre_cvx_hook)

        # 5) Build any selection‐matrices / targets here, then call CVX
        S_nodes, S_edges, v_t, z_t, y_t, f_t, I_t = \
            self.create_selection_matrices_and_targets(N, E, x.device)

        opt_out, v_out = self.cvx_layer(
            yw, vw,
            data.cvx_p_inj.squeeze(0), data.cvx_q_inj.squeeze(0), data.cvx_y0.squeeze(0),
            data.cvx_r_pu.squeeze(0),   data.cvx_x_pu.squeeze(0),
            data.cvx_bigM_flow.squeeze(0), data.cvx_bigM_v.squeeze(0),
            data.cvx_A_from.squeeze(0),    data.cvx_A_to.squeeze(0),
            data.cvx_sub_mask.squeeze(0),  data.cvx_non_sub_mask.squeeze(0),
            data.cvx_bigM_flow_sq.squeeze(0), data.cvx_z_line_sq.squeeze(0),
            S_nodes, S_edges, v_t, z_t, y_t, f_t, I_t
        )

        y_opt       = opt_out.clone().requires_grad_()
        v_sq_opt    = v_out.clone().requires_grad_()

        # now hooks will register
        y_opt.register_hook(lambda g: self._hook_log("Post-CVX y_opt", g))
        v_sq_opt.register_hook(lambda g: self._hook_log("Post-CVX v_sq", g))


        # # now slice off what you need for return
        # y_un = y_opt[:E]
        # v_un = v_sq_opt[:N]

        y_un   = P_e @ y_opt     # shape [E]
        v_un   = P_n @ v_sq_opt  # shape [N]

        return {
            "switch_logits": yw_unp,
            "voltage_scores": vw_unp,
            "switch_predictions": y_un,
            "voltage_predictions": v_un.sqrt(),
        }
        
    def log_warmstart_grads(self, stage="After backward"):
        """Call this *after* loss.backward() to log .grad norms on the warm‐starts."""
        if hasattr(self, 'last_yw') and self.last_yw.grad is not None:
            logger.info(f"{stage} yw.grad norm: {self.last_yw.grad.norm().item():.6f}")
        if hasattr(self, 'last_vw') and self.last_vw.grad is not None:
            logger.info(f"{stage} vw.grad norm: {self.last_vw.grad.norm().item():.6f}")