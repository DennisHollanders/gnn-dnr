import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class PIGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64, 32],latent_dim=None,activation=None, dropout_rate=0.1):
        super(PIGNN, self).__init__()
        self.encoder = GNNEncoder(input_dim, hidden_dims, dropout_rate=dropout_rate)
        emb_dim = hidden_dims[-1]
        self.switch_head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.qubo_layer = DifferentiableQUBO()
        self.kirchhoff_pinn = KirchhoffPINN(emb_dim, dropout_rate=dropout_rate)
        self.radial_pinn = RadialPINN(emb_dim, dropout_rate=dropout_rate)
        self.loadflow_pinn = LoadFlowPINN(emb_dim, dropout_rate=dropout_rate)

    def forward(self, data):
        #print("print of complete data", data)
        node_emb = self.encoder(data.x.to(torch.float32), data.edge_index)
        graph_emb = pyg_nn.global_mean_pool(node_emb, data.batch)
        switch_scores = self.switch_head(graph_emb).squeeze(-1)
        switch_states, qubo_loss = self.qubo_layer(switch_scores, data)
        voltage_pred, kirchhoff_loss = self.kirchhoff_pinn(node_emb, data)
        radial_loss = self.radial_pinn(switch_states, data)
        flow_pred, loadflow_loss = self.loadflow_pinn(node_emb, voltage_pred, data)
        total_physics_loss = kirchhoff_loss + radial_loss + loadflow_loss
        return {
            "switch_scores": switch_states,
            "losses": {
                "qubo_loss": qubo_loss,
                "total_physics_loss": total_physics_loss
            }
        }

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for out_dim in hidden_dims:
            self.layers.append(pyg_nn.GCNConv(in_dim, out_dim))
            in_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, edge_index):
        out = x
        for conv in self.layers:
            residual = out
            out = conv(out, edge_index)
            out = F.relu(out)
            out = self.dropout(out)
            if residual.shape[-1] == out.shape[-1]:
                out = out + residual
        return out

class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_rate=0.1, activation=nn.GELU):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = activation()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.skip = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual
    

class PINNBranchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64, 32], dropout_rate=0.1, activation=nn.GELU):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(ResidualMLPBlock(in_dim, h, dropout_rate, activation))
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 1)
        
    def forward(self, x):
        x = self.mlp(x)
        return self.out(x)


# Differentiable QUBO optimization layer
class DifferentiableQUBO(nn.Module):
    def forward(self, scores, data):
        decisions = torch.sigmoid(scores)
        qubo_loss = self.compute_qubo_loss(decisions, data)
        return decisions, qubo_loss

    def compute_qubo_loss(self, decisions, data):
        decisions_2d = decisions.unsqueeze(0)
        qubo_matrix = torch.eye(decisions_2d.shape[-1], device=decisions.device)
        loss = decisions_2d @ qubo_matrix @ decisions_2d.T
        return loss.mean()

# Kirchhoff PINN branch
class KirchhoffPINN(nn.Module):
    def __init__(self, emb_dim, dropout_rate=0.1):
        super().__init__()
        self.mlp = PINNBranchMLP(emb_dim, hidden_dims=[64, 32], dropout_rate=dropout_rate, activation=nn.GELU)
        
    def forward(self, node_emb, data):
        voltages = self.mlp(node_emb).squeeze(-1)
        kcl_residual = compute_kcl_residual(voltages, data)
        # kvl_residual = compute_kvl_residual(voltages, data)
        loss = kcl_residual.pow(2).mean()  # + kvl_residual.pow(2).mean() 
        return voltages, loss


class RadialPINN(nn.Module):
    # TODO: Possible to add constraint as a message passing layer weighted by switch status?

    def __init__(self, emb_dim, dropout_rate=0.1):
        super().__init__()
        self.mlp = PINNBranchMLP(emb_dim, hidden_dims=[32, 16], dropout_rate=dropout_rate, activation=nn.GELU)
        
    def forward(self, switch_states, data):
        num_nodes = data.num_nodes
        closed_switches = switch_states.sum()
        radial_loss = (closed_switches - (num_nodes - 1))**2 / num_nodes
        return radial_loss

class LoadFlowPINN(nn.Module):
    def __init__(self, emb_dim, dropout_rate=0.1):
        super().__init__()
        self.mlp = PINNBranchMLP(emb_dim, hidden_dims=[64, 32], dropout_rate=dropout_rate, activation=nn.GELU)
        
    def forward(self, node_emb, voltages, data):
        flows = self.mlp(node_emb).squeeze(-1)
        loadflow_residual = compute_loadflow_residual(flows, voltages, data)
        loss = loadflow_residual.pow(2).mean()
        return flows, loss

def compute_kcl_residual(voltages, data):
    G = data.conductance_matrix  
    currents = G @ voltages
    residual = currents - data.x[:, 0]  # p
    return residual

def compute_kvl_residual(voltages, data):
    row, col = data.edge_index
    R = data.edge_attr[:, 0]
    voltage_drop_pred = voltages[row] - voltages[col]
    current_pred = data.line_currents
    voltage_drop_actual = R * current_pred
    residual = voltage_drop_pred - voltage_drop_actual
    return residual

def compute_loadflow_residual(flows, voltages, data):
    row, col = data.edge_index
    Z = data.edge_attr[:, :2]  # impedance (R, X)
    voltage_diff = voltages[row] - voltages[col]
    residual = voltage_diff - (Z[:, 0] * flows[row])
    return residual

class PIGNNCriterion(nn.Module):
    def __init__(self, weight_switch=1.0, weight_physics=10.0):
        super().__init__()
        self.weight_switch = weight_switch
        self.weight_physics = weight_physics
        self.switch_loss_fn = nn.BCEWithLogitsLoss()
        #self.switch_loss_fn = DifferentiableQUBO()

    def forward(self, output, data):
        # Check if the target exists. If not, use a dummy tensor of zeros.
        if hasattr(data, "y") and data.y is not None:
            target = data.y.float()
        else:
            target = torch.zeros_like(output["switch_scores"])
        .
        _, switch_loss = self.switch_loss_fn(output["switch_scores"], target)
        physics_loss = output["losses"]["total_physics_loss"]
        total_loss = self.weight_switch * switch_loss + self.weight_physics * physics_loss

        # Local losses for logging/monitoring
        local_losses = {
            "switch_loss": switch_loss,
            **output["losses"]
        }

        return total_loss, local_losses

