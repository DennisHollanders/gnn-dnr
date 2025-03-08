import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class PIGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16],
                 latent_dim=8, activation='prelu', dropout_rate=0.0):
        super(PIGNN, self).__init__()

        self.encoder = GNNEncoder(input_dim, hidden_dims)

        emb_dim = hidden_dims[-1]

        self.switch_head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.qubo_layer = DifferentiableQUBO()

        # PINN Branches
        self.kirchhoff_pinn = KirchhoffPINN(emb_dim)
        self.radial_pinn = RadialPINN(emb_dim)
        self.loadflow_pinn = LoadFlowPINN(emb_dim)

    def forward(self, data):
        node_emb = self.encoder(data.x.to(torch.float32), data.edge_index)
        graph_emb = pyg_nn.global_mean_pool(node_emb, data.batch)

        # Switch state prediction
        switch_scores = self.switch_head(graph_emb).squeeze(-1)
        switch_states, qubo_loss = self.qubo_layer(switch_scores, data)

        # Pinn branch
        voltage_pred, kirchhoff_loss = self.kirchhoff_pinn(node_emb, data)
        radial_loss = self.radial_pinn(switch_states, data)
        flow_pred, loadflow_loss = self.loadflow_pinn(node_emb, voltage_pred, data)

        total_physics_loss = kirchhoff_loss + radial_loss + loadflow_loss

        return switch_states, qubo_loss, total_physics_loss

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(GNNEncoder, self).__init__()
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(pyg_nn.GCNConv(in_dim, out_dim))
            in_dim = out_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = F.relu(conv(x, edge_index))
        return x

# Differentiable QUBO optimization layer
class DifferentiableQUBO(nn.Module):
    def forward(self, scores, data):
        decisions = torch.sigmoid(scores)
        qubo_loss = self.compute_qubo_loss(decisions, data)
        return decisions, qubo_loss

    def compute_qubo_loss(self, decisions, data):
        qubo_matrix = torch.eye(len(decisions), device=decisions.device)
        loss = decisions @ qubo_matrix @ decisions.T
        return loss.mean()

# Kirchhoff PINN branch
class KirchhoffPINN(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.voltage_head = nn.Sequential(
            nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, node_emb, data):
        voltages = self.voltage_head(node_emb).squeeze(-1)
        kcl_residual = compute_kcl_residual(voltages, data)
        kvl_residual = compute_kvl_residual(voltages, data)
        loss = (kcl_residual.pow(2).mean() + kvl_residual.pow(2).mean())
        return voltages, loss

# Radiality constraint PINN branch
class RadialPINN(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # Radiality is directly enforced on predicted switch states

    def forward(self, switch_states, data):
        num_nodes = data.num_nodes
        closed_switches = switch_states.sum()
        radial_loss = (closed_switches - (num_nodes - 1))**2 / num_nodes
        return radial_loss

# LoadFlow PINN branch
class LoadFlowPINN(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.flow_head = nn.Sequential(
            nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, node_emb, voltages, data):
        flows = self.flow_head(node_emb).squeeze(-1)
        loadflow_residual = compute_loadflow_residual(flows, voltages, data)
        loss = loadflow_residual.pow(2).mean()
        return flows, loss


def compute_kcl_residual(voltages, data):
    G = data.conductance_matrix  
    currents = G @ voltages
    residual = currents - data.net_injection  # (generation - load)
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
    Z = data.edge_attr[:, :2]  # impedance (R,X)
    voltage_diff = voltages[row] - voltages[col]
    residual = voltage_diff - (Z[:,0] * flows)
    return residual

class PIGNNCriterion(nn.Module):
    def __init__(self, weight_switch=1.0, weight_physics=10.0):
        super().__init__()
        self.weight_switch = weight_switch
        self.weight_physics = weight_physics
        #self.switch_loss_fn = nn.BCEWithLogitsLoss()
        self.switch_loss_fn = DifferentiableQUBO()

    def forward(self, output, data):
        switch_loss = self.switch_loss_fn(output["switch_scores"], data.y.float())
        physics_loss = output["losses"]["total_physics_loss"]
        total_loss = self.weight_switch * switch_loss + self.weight_physics * physics_loss

        # Local losses for logging/monitoring
        local_losses = {
            "switch_loss": switch_loss,
            **output["losses"]
        }

        return total_loss, local_losses

