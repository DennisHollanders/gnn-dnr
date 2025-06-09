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

def build_cvx_layer(max_n: int, max_e: int):
    """
    Builds a fully parametric and DPP-compliant CvxpyLayer.

    Args:
        max_n: The maximum number of nodes for padding.
        max_e: The maximum number of edges for padding.
    """
    logger.info(f"Building DPP-compliant CVX layer for size: max_nodes={max_n}, max_edges={max_e}")

    # --- Define CVXPY Variables ---
    v_sq   = cp.Variable(max_n, name='v_sq', nonneg=True)
    p_flow = cp.Variable(max_e, name='p_flow')
    q_flow = cp.Variable(max_e, name='q_flow')
    I_sq   = cp.Variable(max_e, name='I_sq', nonneg=True)
    y_line = cp.Variable(max_e, name='y_line')
    z_bus  = cp.Variable(max_n, name='z_bus')

    # --- Define CVXPY Parameters ---
    # Warm starts from GNN
    y_warm = cp.Parameter(max_e, name='y_warm')
    v_warm = cp.Parameter(max_n, name='v_warm')

    # Graph-specific parameters
    p_inj        = cp.Parameter(max_n, name='p_inj')
    q_inj        = cp.Parameter(max_n, name='q_inj')
    y0           = cp.Parameter(max_e, name='y0')
    r_pu         = cp.Parameter(max_e, name='r_pu')
    x_pu         = cp.Parameter(max_e, name='x_pu')
    bigM_flow    = cp.Parameter(max_e, name='bigM_flow')
    bigM_v_param = cp.Parameter(nonneg=True, name='bigM_v')

    # Parameters for topology, masks, and pre-calculated values
    A_from       = cp.Parameter((max_e, max_n), name='A_from')
    A_to         = cp.Parameter((max_e, max_n), name='A_to')
    sub_mask     = cp.Parameter(max_n, name='sub_mask', nonneg=True)
    non_sub_mask = cp.Parameter(max_n, name='non_sub_mask', nonneg=True)
    bigM_flow_sq = cp.Parameter(max_e, name='bigM_flow_sq', nonneg=True)
    z_line_sq    = cp.Parameter(max_e, name='z_line_sq', nonneg=True)

    v_lower_sq, v_upper_sq = 0.9**2, 1.1**2

    # --- Build DPP-Compliant Constraints ---
    constraints = []

    # Relaxed binary variable bounds [0, 1]
    constraints += [0 <= y_line, y_line <= 1]
    constraints += [0 <= z_bus,  z_bus  <= 1]

    # Substation constraints (vectorized)
    constraints += [cp.multiply(sub_mask, v_sq) == sub_mask] # Fixes v_sq to 1 for substations
    constraints += [cp.multiply(sub_mask, z_bus) == sub_mask] # Fixes z_bus to 1 for substations

    # Voltage bounds for non-substation buses (vectorized)
    constraints += [cp.multiply(non_sub_mask, v_sq) >= cp.multiply(non_sub_mask, v_lower_sq * z_bus)]
    constraints += [cp.multiply(non_sub_mask, v_sq) <= cp.multiply(non_sub_mask, (v_upper_sq + bigM_v_param * (1 - z_bus)))]

    # Line energization depends on connected buses being energized
    constraints += [y_line <= A_from @ z_bus]
    constraints += [y_line <= A_to @ z_bus]

    # Power flow and current bounds (Big-M)
    constraints += [cp.abs(p_flow) <= cp.multiply(bigM_flow, y_line)]
    constraints += [cp.abs(q_flow) <= cp.multiply(bigM_flow, y_line)]
    constraints += [I_sq <= cp.multiply(bigM_flow_sq, y_line)] # Use pre-squared M

    # Rotated Second-Order Cone constraint for power flow
    # P^2 + Q^2 <= V_sq * I_sq, which is equivalent to ||[2P, 2Q, V_sq-I_sq]|| <= V_sq+I_sq
    # No relaxation needed as flows are already forced to 0 by the bounds above.
    v_from = A_from @ v_sq
    constraints += [
        cp.norm(cp.hstack([2 * p_flow, 2 * q_flow, v_from - I_sq]), p=2, axis=1)
        <= v_from + I_sq
    ]

    # Voltage drop constraint (vectorized)
    voltage_drop_vec = (A_to @ v_sq - A_from @ v_sq
                    + 2 * (cp.multiply(r_pu, p_flow) + cp.multiply(x_pu, q_flow))
                    - cp.multiply(z_line_sq, I_sq))
    constraints += [cp.abs(voltage_drop_vec) <= cp.multiply(bigM_v_param, (1 - y_line))]

    # Power balance for non-substation buses (vectorized)
    p_balance = (A_from - A_to).T @ p_flow
    q_balance = (A_from - A_to).T @ q_flow
    constraints += [cp.multiply(non_sub_mask, p_balance) == cp.multiply(non_sub_mask, cp.multiply(p_inj, z_bus))]
    constraints += [cp.multiply(non_sub_mask, q_balance) == cp.multiply(non_sub_mask, cp.multiply(q_inj, z_bus))]

    # Radiality constraint (spanning tree)
    constraints += [cp.sum(y_line) == cp.sum(z_bus) - 1]

    # --- Objective Function ---
    loss_objective = cp.sum(cp.multiply(r_pu, I_sq)) # Minimize resistive losses
    loss_objective += 0.001 * cp.sum_squares(y_line - y_warm) # Warm-start for switches
    loss_objective += 0.001 * cp.sum_squares(v_sq - v_warm)   # Warm-start for voltage
    loss_objective += 0.0001 * cp.norm(y_line - y0, 1)        # Penalty for changing switch status

    objective = cp.Minimize(loss_objective)
    problem = cp.Problem(objective, constraints)

    # Verify DPP compliance
    assert problem.is_dcp(dpp=True), "Problem is not DPP-compliant!"
    logger.info("CVXPY problem is successfully verified as DPP-compliant.")

    # Define the layer with the full list of parameters
    cvx_layer = CvxpyLayer(
        problem,
        parameters=[
            y_warm, v_warm, p_inj, q_inj, y0, r_pu, x_pu, bigM_flow,
            bigM_v_param, A_from, A_to, sub_mask, non_sub_mask,
            bigM_flow_sq, z_line_sq
        ],
        variables=[y_line, v_sq]
    )
    return cvx_layer
class cvx(nn.Module):
    def __init__(self, **kwargs):
        super(cvx, self).__init__()
        hidden_dims = kwargs['hidden_dims']
        latent_dim = kwargs['latent_dim']

        # Store max_n and max_e and build the layer during initialization
        self.max_n = kwargs['max_n']
        self.max_e = kwargs['max_e']
        self.cvx_layer = build_cvx_layer(self.max_n, self.max_e)

        # --- GNN layers ---
        self.node_encoder = nn.Linear(kwargs['node_input_dim'], hidden_dims[0])
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)
        ])
        self.gnn_layers.append(GCNConv(hidden_dims[-1], latent_dim))

        # --- Predictor heads for warm-starts ---
        self.switch_predictor = nn.Sequential(nn.Linear(latent_dim * 2, 1), nn.Sigmoid())
        self.voltage_predictor = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=kwargs['dropout_rate'])
        self.v_lower = 0.9
        self.v_upper = 1.1
        self.logger = logging.getLogger(__name__) # Add logger

    def forward(self, data):
        # The GNN part remains the same
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.node_encoder(x))
        for conv in self.gnn_layers:
            x = self.relu(conv(x, edge_index))
        edge_emb = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)

        # Predict warm-starts
        y_warm_pred = self.switch_predictor(edge_emb).squeeze(-1)
        v_raw_pred  = self.voltage_predictor(x).squeeze(-1)
        v_warm_pred = (self.v_lower + (self.v_upper - self.v_lower) * v_raw_pred).pow(2)

        # The CVX layer expects batched inputs.
        # We assume the dataloader handles batching of all cvx_* tensors.
        y_warm = y_warm_pred
        v_warm = v_warm_pred

        # Solve the optimization problem using the CvxpyLayer
        # Pass all required parameters in the correct order
        y_opt, v_sq_opt = self.cvx_layer(
            y_warm, v_warm,
            data.cvx_p_inj, data.cvx_q_inj, data.cvx_y0,
            data.cvx_r_pu, data.cvx_x_pu, data.cvx_bigM_flow,
            data.cvx_bigM_v, data.cvx_A_from, data.cvx_A_to,
            data.cvx_sub_mask, data.cvx_non_sub_mask,
            data.cvx_bigM_flow_sq, data.cvx_z_line_sq,
            solver_args={'verbose': False, 'solve_method': 'SCS'}
        )

        return y_opt, v_sq_opt