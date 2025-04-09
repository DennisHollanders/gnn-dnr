import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

# Baseline MLP using node features only
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64, 32], latent_dim=None, activation=F.relu, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for out_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(in_dim, latent_dim)
        self.activation = activation
        self.batch_norm = nn.BatchNorm1d(latent_dim)

    def forward(self, data):
        x = data.x  # node features
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.batch_norm(x)
        x = self.out(x)
        return x

# Extended MLP that also uses edge features
class MLPWithEdge(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dims=[64, 64, 32], latent_dim=None, activation=F.relu, dropout_rate=0.1):
        super(MLPWithEdge, self).__init__()
        # Combined input: node features concatenated with aggregated edge features.
        combined_input_dim = node_input_dim + edge_input_dim
        self.layers = nn.ModuleList()
        in_dim = combined_input_dim
        for out_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(in_dim, latent_dim)
        self.activation = activation
        self.batch_norm = nn.BatchNorm1d(latent_dim)

    def forward(self, data):
        x = data.x  # [num_nodes, node_input_dim]
        # Aggregate edge features (assuming data.edge_index and data.edge_attr exist)
        # Here we use the source node (data.edge_index[0]) for aggregation.
        edge_agg = scatter_mean(data.edge_attr, data.edge_index[0], dim=0, dim_size=x.size(0))
        # Concatenate node features with the aggregated edge features
        x = torch.cat([x, edge_agg], dim=-1)
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.batch_norm(x)
        x = self.out(x)
        return x
