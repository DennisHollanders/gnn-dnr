import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np

class cvx(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim,
                 hidden_dims=[64, 32, 16], latent_dim=8, activation='relu', dropout_rate=0.0):

        super().__init__()

        # 1) Node encoder: raw node features → hidden_dims[0]
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(p=0.0)
        )

        # 2) Edge encoder: raw edge features → hidden_dims[0]
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(p=0.0)
        )

        # 3) GCNConv layers
        gnn_layers = []
        for i in range(len(hidden_dims)):
            in_ch  = hidden_dims[i]
            out_ch = hidden_dims[i+1] if i+1 < len(hidden_dims) else latent_dim
            gnn_layers.append(GCNConv(in_ch, out_ch))
        self.gnn_layers = nn.ModuleList(gnn_layers)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.0)

        # 4) Warm‐start predictors
        self.switch_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )
        self.voltage_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )

        # 5) Placeholder for the CVX layer; will be rebuilt each forward
        self.cvx_layer = None
        self.v_lower = 0.9
        self.v_upper = 1.1

    def _build_cvx_template(self, N_true, E_true,
                            from_idx_list, to_idx_list,
                            r_pu_list, x_pu_list,
                            bigM_flow_list, bigM_v_scalar,
                            sub_idx_list):

        # Convert to Python lists if needed (they must be plain ints/floats):
        from_idx = [int(x) for x in from_idx_list]       # length = E_true
        to_idx   = [int(x) for x in to_idx_list]         # length = E_true
        r_pu     = [float(x) for x in r_pu_list]         # length = E_true
        x_pu     = [float(x) for x in x_pu_list]         # length = E_true
        bigM_flow= [float(x) for x in bigM_flow_list]    # length = E_true
        bigM_v   = float(bigM_v_scalar)                  # single scalar
        sub_idx  = [int(x) for x in sub_idx_list]        # length = N_true; each 0 or 1

        # ---------- CVX variables ----------
        v_sq   = cp.Variable(N_true,   nonneg=True, name='v_sq')    # squared voltage
        p_flow = cp.Variable(E_true,   name='p_flow')                # P‐flow
        q_flow = cp.Variable(E_true,   name='q_flow')                # Q‐flow
        I_sq   = cp.Variable(E_true,   nonneg=True, name='I_sq')     # I²
        y_line = cp.Variable(E_true,   name='y_line')                # switch relaxed
        z_bus  = cp.Variable(N_true,   name='z_bus')                 # bus on/off

        # ---------- CVX parameters (the only “time‐varying” parts) ----------
        y_warm = cp.Parameter(E_true, name='y_warm')          # warm‐start switches
        v_warm = cp.Parameter(N_true, nonneg=True, name='v_warm')  # warm‐start voltages
        p_inj  = cp.Parameter(N_true, name='p_inj')           # real injections
        q_inj  = cp.Parameter(N_true, name='q_inj')           # reactive injections
        y0     = cp.Parameter(E_true, name='y0')              # previous switch states

        constraints = []

        # 1) Box constraints on y_line and z_bus
        constraints += [
            0 <= y_line,  y_line <= 1,
            0 <= z_bus,   z_bus  <= 1
        ]

        # 2) Substation vs non‐substation constraints
        for i in range(N_true):
            if sub_idx[i] == 1:
                # If bus i is a substation, fix v_sq[i] == 1 and z_bus[i] == 1
                constraints.append(v_sq[i] == 1.0)
                constraints.append(z_bus[i] == 1.0)
            else:
                # Otherwise, apply big‐M voltage bounds:
                constraints.append(v_sq[i] <= self.v_upper**2)
                constraints.append(v_sq[i] >= (self.v_lower**2)*z_bus[i])

        # 3) Big‐M P/Q/I² constraints
        for e in range(E_true):
            M_e = bigM_flow[e]
            constraints += [
                p_flow[e] <=  M_e * y_line[e],
                p_flow[e] >= -M_e * y_line[e],
                q_flow[e] <=  M_e * y_line[e],
                q_flow[e] >= -M_e * y_line[e],
                I_sq[e]   <= (M_e**2) * y_line[e]
            ]

        # 4) SOC‐cone constraints
        for e in range(E_true):
            u = from_idx[e]  # integer index of up‐bus
            M_e = bigM_flow[e]
            constraints.append(
                cp.norm(cp.vstack([
                    2 * p_flow[e],
                    2 * q_flow[e],
                    v_sq[u] - I_sq[e]
                ]), 2)
                <= v_sq[u] + I_sq[e] + 2 * (M_e**2) * (1 - y_line[e])
            )

        # 5) Voltage‐drop constraints
        for e in range(E_true):
            u = from_idx[e]
            v = to_idx[e]
            expr = (
                v_sq[v] - v_sq[u]
                + 2 * (r_pu[e]*p_flow[e] + x_pu[e]*q_flow[e])
                - (r_pu[e]**2 + x_pu[e]**2)*I_sq[e]
            )
            constraints += [
                expr <= bigM_v * (1 - y_line[e]),
                expr >= -bigM_v * (1 - y_line[e])
            ]

        # 6) Line‐bus linking
        for e in range(E_true):
            constraints += [
                y_line[e] <= z_bus[from_idx[e]],
                y_line[e] <= z_bus[to_idx[e]]
            ]

        # 7) Power balance at non‐substation buses
        for i in range(N_true):
            p_out_list = [p_flow[e] for e in range(E_true) if from_idx[e] == i]
            p_in_list  = [p_flow[e] for e in range(E_true) if to_idx[e]   == i]
            q_out_list = [q_flow[e] for e in range(E_true) if from_idx[e] == i]
            q_in_list  = [q_flow[e] for e in range(E_true) if to_idx[e]   == i]

            p_out = cp.sum(p_out_list) if p_out_list else 0
            p_in  = cp.sum(p_in_list)  if p_in_list  else 0
            q_out = cp.sum(q_out_list) if q_out_list else 0
            q_in  = cp.sum(q_in_list)  if q_in_list  else 0

            constraints.append(p_out - p_in == p_inj[i] * z_bus[i])
            constraints.append(q_out - q_in == q_inj[i] * z_bus[i])

        # 8) Objective = I²·R + switch penalty + warm‐start penalty
        loss = cp.sum(cp.multiply(r_pu, I_sq))
        loss += 0.0001 * cp.sum(cp.abs(y_line - y0))       # switch‐change penalty
        loss += 0.01   * cp.sum_squares(y_line - y_warm)   # warm‐start on switches
        loss += 0.01   * cp.sum_squares(v_sq   - v_warm)   # warm‐start on voltages

        problem = cp.Problem(cp.Minimize(loss), constraints)

        # Only the five parameters change each solve: [y_warm, v_warm, p_inj, q_inj, y0]
        self.cvx_layer = CvxpyLayer(
            problem,
            parameters=[y_warm, v_warm, p_inj, q_inj, y0],
            variables=[y_line, v_sq]
        )

        #print("Is DPP? ", problem.is_dcp(dpp=True))
        #print("Is DCP? ", problem.is_dcp(dpp=False))


    def forward(self, data):
        # === 1) Run GNN to get “warm-start” predictions ===
        x     = data.x                   
        e_idx = data.edge_index          

        # 1.1) Node encoder
        x = self.node_encoder(x)         
        x = self.relu(x)
        x = self.dropout(x)

        # 1.2) GCNConv stack
        for i, conv in enumerate(self.gnn_layers):
            x = conv(x, e_idx)
            if i < len(self.gnn_layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        # At this point: x.shape == [total_nodes, latent_dim]

        # 1.3) Edge warm-start (concatenate “from” and “to” embeddings)
        u_emb = x[e_idx[0]]              # [total_edges, latent_dim]
        v_emb = x[e_idx[1]]              # [total_edges, latent_dim]
        edge_emb = torch.cat([u_emb, v_emb], dim=1)  # [total_edges, latent_dim*2]
        switch_scores = self.switch_predictor(edge_emb).squeeze(-1)

        # 1.4) Node warm-start (voltage predictions)
        voltage_scores = self.voltage_predictor(x).squeeze(-1)
        v_pred_scaled  = self.v_lower + (self.v_upper - self.v_lower) * voltage_scores
        v_sq_pred      = v_pred_scaled.pow(2)  

        # === 2) Rebuild CVX template for exactly this graph (batch_size=1) ===
        N_true = data.num_nodes                   # actual number of nodes
        E_true = data.edge_index.shape[1]         # actual number of edges

        # Extract padded CVX features (as NumPy on CPU)
        from_idx_full_padded  = data.cvx_from_idx.cpu().numpy().reshape(-1)   # length = max_E
        to_idx_full_padded    = data.cvx_to_idx.cpu().numpy().reshape(-1)     # length = max_E
        r_pu_full_padded      = data.cvx_r_pu.cpu().numpy().reshape(-1)       # length = max_E
        x_pu_full_padded      = data.cvx_x_pu.cpu().numpy().reshape(-1)       # length = max_E
        bigM_flow_full_padded = data.cvx_bigM_flow.cpu().numpy().reshape(-1)  # length = max_E
        sub_idx_full_padded   = data.cvx_sub_idx.cpu().numpy().reshape(-1)    # length = max_N
        bigM_v_scalar         = float(data.cvx_bigM_v[0].item())              # scalar

        # Slice out only the “true” entries:
        from_idx_np   = from_idx_full_padded[:E_true].astype(int).tolist()
        to_idx_np     = to_idx_full_padded[:E_true].astype(int).tolist()
        r_pu_np       = r_pu_full_padded[:E_true].astype(float).tolist()
        x_pu_np       = x_pu_full_padded[:E_true].astype(float).tolist()
        bigM_flow_np  = bigM_flow_full_padded[:E_true].astype(float).tolist()
        sub_idx_np    = sub_idx_full_padded[:N_true].astype(int).tolist()

        # Rebuild the CVX problem on exactly (N_true, E_true)
        self._build_cvx_template(
            N_true, E_true,
            from_idx_np, to_idx_np,
            r_pu_np,   x_pu_np,
            bigM_flow_np, bigM_v_scalar,
            sub_idx_np
        )

        y_warm_t = switch_scores[:E_true].cpu()    
        v_warm_t = v_sq_pred[:N_true].cpu()  

        # Extract true p_inj, q_inj, and y0, but as CPU‐tensors that do not require grad
        p_inj_full_padded = data.cvx_p_inj.cpu().numpy().reshape(-1)  # (max_N,)
        q_inj_full_padded = data.cvx_q_inj.cpu().numpy().reshape(-1)  # (max_N,)
        y0_full_padded    = data.cvx_y0.cpu().numpy().reshape(-1)     # (max_E,)

        p_inj_np = p_inj_full_padded[:N_true].astype(np.float32)
        q_inj_np = q_inj_full_padded[:N_true].astype(np.float32)
        y0_np    = y0_full_padded[:E_true].astype(np.float32)

        p_inj_t = torch.from_numpy(p_inj_np)   # CPU tensor, requires_grad=False
        q_inj_t = torch.from_numpy(q_inj_np)
        y0_t    = torch.from_numpy(y0_np)

        # === 4) Call CvxpyLayer with (y_warm_t, v_warm_t, p_inj_t, q_inj_t, y0_t) ===
        y_opt_full, v_sq_opt_full = self.cvx_layer(
            y_warm_t, v_warm_t,
            p_inj_t,  q_inj_t, y0_t
        )

        # print(f"CVX solution: y_opt_full.requires_grad = {y_opt_full.requires_grad}")
        # print(f"CVX solution: y_opt_full.grad_fn       = {y_opt_full.grad_fn}")

        return {
            "switch_predictions": switch_scores,  # [total_edges], requires_grad=True
            "voltage_predictions": v_sq_pred,     # [total_nodes], requires_grad=True
            "switch_scores":      y_opt_full,     # [E_true], requires_grad=True
            "voltage_optimal":    v_sq_opt_full   # [N_true], requires_grad=True
        }

    def get_edge_embeddings(self, node_emb, edge_index):
        u = node_emb[edge_index[0]]
        v = node_emb[edge_index[1]]
        return torch.cat([u, v], dim=1)
