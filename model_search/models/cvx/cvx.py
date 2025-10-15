import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import logging
from concurrent.futures import ThreadPoolExecutor
import sys 
from pathlib import Path
import os 

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "model_search")])

from AdvancedMLP.AdvancedMLP import AdvancedMLP

logger = logging.getLogger(__name__)

def build_cvx_layer(max_n: int, max_e: int):
    """Build DPP-compliant CVX layer using affine constraints only"""
    # Full-size variables
    v_sq         = cp.Variable(max_n, nonneg=True)
    p_flow       = cp.Variable(max_e)
    q_flow       = cp.Variable(max_e)
    I_sq         = cp.Variable(max_e, nonneg=True)
    y_line       = cp.Variable(max_e, nonneg=True)
    z_bus        = cp.Variable(max_n, nonneg=True)
    f            = cp.Variable(max_e, nonneg=True)
    s            = cp.Variable(nonneg=True)

    # Warm start parameters 
    y_warm       = cp.Parameter(max_e, nonneg=True)
    v_warm       = cp.Parameter(max_n, nonneg=True)
    
    # CVX parameters 
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
    
    # DPP-compliant masking using selection matrices 
    S_nodes      = cp.Parameter((max_n, max_n), nonneg=True)  
    S_edges      = cp.Parameter((max_e, max_e), nonneg=True)  
    
    # Target values for inactive elements
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
    
    # DPP-compliant using affine constraints
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

    # Add a new parameter for the penalty weight
    lambda_penalty = cp.Parameter(nonneg=True)

    # Objective function
    loss  = cp.sum(cp.multiply(r_pu_full, I_sq))
    loss += 0.001*cp.sum_squares(y_line - y_warm)
    loss += 0.001*cp.sum_squares(v_sq - v_warm)
    loss += 0.01*cp.norm1(y_line - y0_full)


    penalty = -cp.sum(y_line - cp.square(y_line)) 
    loss += lambda_penalty * penalty
    cons += [cp.sum(y_line) == cp.sum(z_bus) - 1]


    problem = cp.Problem(cp.Minimize(loss), cons)

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
            v_target, z_target, y_target, flow_target, I_target,
            lambda_penalty 
        ],
        variables=[y_line, v_sq],
    )

class cvx(nn.Module):
    def __init__(self, **K,):
        super().__init__()
        self.max_n = K['max_n']
        self.max_e = K['max_e']
        self.cvx_layer = K["cvx_layer"] 
        self.lambda_penalty = 2
     
        self.vl, self.vu = 0.9, 1.1
        
        self.gnn = AdvancedMLP(**K)
        
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

    def forward(self, data):
        # 1) GNN forward pass
        gnn_out = self.gnn(data)
        
        all_switch = gnn_out["switch_logits"]
        all_voltage = gnn_out.get("node_v", gnn_out.get("voltage_predictions"))
        
        # 2) Get actual edge and node counts
        ptr = data.ptr.tolist()
        node_counts = [ptr[i+1] - ptr[i] for i in range(len(ptr)-1)]
        
        edge_batch = data.batch[data.edge_index[0]]
        num_graphs = len(node_counts)
        edge_counts = []
        
        for i in range(num_graphs):
            edges_in_graph_i = (edge_batch == i).sum().item()
            edge_counts.append(edges_in_graph_i)
        
        # 3) Split based on actual counts
        switch_splits = torch.split(all_switch, edge_counts)
        voltage_splits = torch.split(all_voltage, node_counts)
        
        # 4) Prepare CVX arguments
        cvx_args = []
        for i, (n_e_actual, n_n_actual) in enumerate(zip(edge_counts, node_counts)):
            sw_i = switch_splits[i]
            vw_i = voltage_splits[i]
            
            n_e_padded = data.cvx_E[i].item()
            n_n_padded = data.cvx_N[i].item()
            
            pad_e = n_e_padded - n_e_actual
            pad_n = n_n_padded - n_n_actual
            
            # Apply sigmoid to get  probabilities in [0, 1]
            sw_probs = torch.sigmoid(sw_i)
            sw_full = torch.cat([sw_probs, torch.zeros(pad_e, device=sw_probs.device)], dim=0)\
                        .clone().requires_grad_()
            
            # attempt with clamp volt
            vw_clamped = torch.clamp(vw_i, 0.95, 1.05)
            vw_full = torch.cat([vw_clamped, torch.ones(pad_n, device=vw_i.device)],
                                dim=0).clone().requires_grad_()
            
            S_nodes, S_edges, v_t, z_t, y_t, f_t, I_t = \
                self.create_selection_matrices_and_targets(n_n_actual, n_e_actual,
                                                        sw_full.device)
            
            params = [
                sw_full, vw_full,
                data.cvx_p_inj[i],
                data.cvx_q_inj[i],
                data.cvx_y0[i],
                data.cvx_r_pu[i],
                data.cvx_x_pu[i],
                data.cvx_bigM_flow[i],
                data.cvx_bigM_v[i].squeeze(),
                data.cvx_A_from[i],
                data.cvx_A_to[i],
                data.cvx_sub_mask[i],
                data.cvx_non_sub_mask[i],
                data.cvx_bigM_flow_sq[i],
                data.cvx_z_line_sq[i],
                S_nodes, S_edges,
                v_t, z_t, y_t, f_t, I_t,
                torch.tensor(self.lambda_penalty,
                        dtype=sw_full.dtype,
                        device=sw_full.device)
            ]
            cvx_args.append(params)
        
        # 5) Parallel CVX solves with simple try/except
        def _solve(args):
            try:
                y_opt, v_sq_opt = self.cvx_layer(*args)
                return y_opt, v_sq_opt, True  
            except Exception as e:
                y_fallback = args[0]  
                v_fallback_sq = args[1] ** 2  
                
                if "infeasible" in str(e).lower():
                    logger.debug(f"CVX infeasible, using warm start values")
                else:
                    logger.warning(f"CVX error: {e}, using warm start values")
                
                return y_fallback, v_fallback_sq, False  
        
        max_workers = min(len(cvx_args), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            results = list(exe.map(_solve, cvx_args))
        
        # Track success rate
        successes = sum(1 for _, _, success in results if success)
        if successes < len(results):
            logger.info(f"CVX solve rate: {successes}/{len(results)} successful")
        
        # 6) Stitch outputs back
        switch_out = []
        voltage_out = []
        for (y_full, v_sq_full, _), n_e_actual, n_n_actual in zip(results,
                                                                edge_counts,
                                                                node_counts):
            switch_out.append(y_full[:n_e_actual])
            voltage_out.append(torch.sqrt(torch.clamp(v_sq_full[:n_n_actual], min=0.0)))
        
        switch_preds = torch.cat(switch_out, dim=0)
        voltage_preds = torch.cat(voltage_out, dim=0)
        
        switch_preds = torch.clamp(switch_preds, 0.0, 1.0)
        
        return {
            **gnn_out,
            "switch_predictions": switch_preds,
            "voltage_predictions": voltage_preds,
        }
    def log_warmstart_grads(self, stage="After backward"):
        """Call this *after* loss.backward() to log .grad norms on the warm‐starts."""
        if hasattr(self, 'last_yw') and self.last_yw.grad is not None:
            logger.info(f"{stage} yw.grad norm: {self.last_yw.grad.norm().item():.6f}")
        if hasattr(self, 'last_vw') and self.last_vw.grad is not None:
            logger.info(f"{stage} vw.grad norm: {self.last_vw.grad.norm().item():.6f}")
