import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

def build_cvx_layer(max_n: int, max_e: int):
    v_sq         = cp.Variable(max_n, nonneg=True)
    p_flow       = cp.Variable(max_e)
    q_flow       = cp.Variable(max_e)
    I_sq         = cp.Variable(max_e, nonneg=True)
    y_line       = cp.Variable(max_e)
    z_bus        = cp.Variable(max_n)

    y_warm       = cp.Parameter(max_e)
    v_warm       = cp.Parameter(max_n)

    p_inj        = cp.Parameter(max_n)
    q_inj        = cp.Parameter(max_n)
    y0           = cp.Parameter(max_e)
    r_pu         = cp.Parameter(max_e)
    x_pu         = cp.Parameter(max_e)
    bigM_flow    = cp.Parameter(max_e, nonneg=True)
    bigM_v  = cp.Parameter(nonneg=True)

    A_from       = cp.Parameter((max_e, max_n))
    A_to         = cp.Parameter((max_e, max_n))
    sub_mask     = cp.Parameter(max_n, nonneg=True)
    non_sub_mask = cp.Parameter(max_n, nonneg=True)
    bigM_flow_sq = cp.Parameter(max_e, nonneg=True)
    z_line_sq    = cp.Parameter(max_e, nonneg=True)

    v_low_sq  = 0.9**2
    v_high_sq = 1.1**2

    cons = []
    cons += [y_line >= 0, y_line <= 1]
    cons += [z_bus  >= 0, z_bus  <= 1]
    cons += [cp.multiply(sub_mask, v_sq)  == sub_mask]
    cons += [cp.multiply(sub_mask, z_bus) == sub_mask]

    mask_z = cp.multiply(non_sub_mask, z_bus)
    cons += [cp.multiply(non_sub_mask, v_sq) >= cp.multiply(mask_z, v_low_sq)]
    cons += [
        cp.multiply(non_sub_mask, v_sq)
        <= cp.multiply(non_sub_mask, v_high_sq)
         + cp.multiply(bigM_v, 1 - z_bus)
    ]

    cons += [y_line <= A_from @ z_bus, y_line <= A_to @ z_bus]

    cons += [
        p_flow <= cp.multiply(bigM_flow, y_line),
       -p_flow <= cp.multiply(bigM_flow, y_line),
        q_flow <= cp.multiply(bigM_flow, y_line),
       -q_flow <= cp.multiply(bigM_flow, y_line),
        I_sq   <= cp.multiply(bigM_flow_sq, y_line)
    ]

    v_from = A_from @ v_sq
    # The SOC constraint must be vectorized to be DPP-compliant.
    soc_X = cp.vstack([
        2 * p_flow,
        2 * q_flow,
        v_from - I_sq
    ])
    cons += [cp.SOC(v_from + I_sq, soc_X)]

    vd = (A_to @ v_sq - v_from
         + 2*(cp.multiply(r_pu, p_flow) + cp.multiply(x_pu, q_flow))
         - cp.multiply(z_line_sq, I_sq))
    cons += [
        vd <= cp.multiply(bigM_v, 1 - y_line),
        vd >= cp.multiply(bigM_v, 1 - y_line)
    ]

    pb = (A_from - A_to).T @ p_flow
    qb = (A_from - A_to).T @ q_flow
    pe = pb - cp.multiply(p_inj, z_bus)
    qe = qb - cp.multiply(q_inj, z_bus)
    Mbal = 1e4
    cons += [
        pe <=  Mbal*sub_mask, -pe <=  Mbal*sub_mask,
        qe <=  Mbal*sub_mask, -qe <=  Mbal*sub_mask
    ]

    cons += [cp.sum(y_line) == cp.sum(z_bus) - 1]

    bad = find_bad_cons(cons)
    print("First non-DPP constraint:", bad)

    loss = cp.sum(cp.multiply(r_pu, I_sq))
    loss += 0.001*cp.sum_squares(y_line - y_warm)
    loss += 0.001*cp.sum_squares(v_sq   - v_warm)
    loss += 0.0001*cp.norm1(y_line - y0)

    problem = cp.Problem(cp.Minimize(loss), cons)
    #assert problem.is_dcp(dpp=True)

    return CvxpyLayer(
        problem,
        parameters=[
            y_warm, v_warm,
            p_inj, q_inj, y0, r_pu, x_pu,
            bigM_flow, bigM_v,
            A_from, A_to, sub_mask, non_sub_mask,
            bigM_flow_sq, z_line_sq
        ],
        variables=[y_line, v_sq]
    )

class cvx(nn.Module):
    def __init__(self, **K):
        super().__init__()
        self.max_n     = K['max_n']
        self.max_e     = K['max_e']
        self.cvx_layer = build_cvx_layer(self.max_n, self.max_e)

        dims = K['hidden_dims']
        L    = K['latent_dim']
        self.node_enc  = nn.Linear(K['node_input_dim'], dims[0])
        self.gnns      = nn.ModuleList([GCNConv(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.gnns.append(GCNConv(dims[-1], L))
        self.sw_pred   = nn.Sequential(nn.Linear(2*L, 1), nn.Sigmoid())
        self.v_pred    = nn.Sequential(nn.Linear(L,   1), nn.Sigmoid())
        self.relu      = nn.ReLU()
        self.drop      = nn.Dropout(p=K['dropout_rate'])
        self.vl, self.vu = 0.9, 1.1

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = self.relu(self.node_enc(x))
        for g in self.gnns:
            x = self.relu(g(x, ei))
        emb = torch.cat([x[ei[0]], x[ei[1]]], dim=1)
        yw  = self.sw_pred(emb).squeeze(-1)
        vr  = self.v_pred(x).squeeze(-1)
        vw  = (self.vl + (self.vu - self.vl)*vr).pow(2)

        y_opt, v_sq_opt = self.cvx_layer(
            yw, vw,
            data.cvx_p_inj, data.cvx_q_inj, data.cvx_y0,
            data.cvx_r_pu, data.cvx_x_pu,
            data.cvx_bigM_flow, data.cvx_bigM_v,
            data.cvx_A_from, data.cvx_A_to,
            data.cvx_sub_mask, data.cvx_non_sub_mask,
            data.cvx_bigM_flow_sq, data.cvx_z_line_sq,
            solver_args={'verbose': True, 'solve_method': 'SCS'}
        )
        result_dict=  {
            "switch_scores": y_opt,
            "voltage_scores": v_sq_opt,
            "switch_predictions": yw,
            "voltage_predictions": vw,
        }
        return result_dict


def find_bad_cons(cons_list):
    if not cp.Problem(cp.Minimize(0), cons_list).is_dcp(dpp=True):
        if len(cons_list) == 1:
            return cons_list[0]
        mid = len(cons_list)//2
        left_bad  = find_bad_cons(cons_list[:mid])
        if left_bad is not None:
            return left_bad
        return find_bad_cons(cons_list[mid:])
    return None
