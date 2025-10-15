import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, global_add_pool

# --- Helper function for activation ---
def get_activation(name):
    """Helper function to get activation layer by name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "selu":
        return nn.SELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation function: {name}")


class MLP(nn.Module):
    def __init__(self, **kwargs):

        super(MLP, self).__init__()
        print("Initializing MLP model with the following parameters:")
        print(kwargs)
        self.activation = get_activation(kwargs["activation"])
        self.dropout = nn.Dropout(kwargs["dropout_rate"])

        # MLP for processing node features
        self.node_mlp_layers = nn.ModuleList()
        in_dim_node = kwargs["node_input_dim"] 
        for out_dim in kwargs["hidden_dims"]:
            self.node_mlp_layers.append(nn.Linear(in_dim_node, out_dim))
            in_dim_node = out_dim
        self.node_mlp_out_dim = in_dim_node 

        # MLP for processing edge features
        self.edge_mlp_layers = nn.ModuleList()
        in_dim_edge = kwargs["edge_input_dim"]
        for out_dim in kwargs["hidden_dims"]: 
            self.edge_mlp_layers.append(nn.Linear(in_dim_edge, out_dim))
            in_dim_edge = out_dim
        self.edge_mlp_out_dim = in_dim_edge 
        combined_dim = self.node_mlp_out_dim * 2 + self.edge_mlp_out_dim
        self.prediction_layer = nn.Linear(combined_dim, 1)
        
        # Batch normalization layers
        self.node_batch_norm = nn.BatchNorm1d(self.node_mlp_out_dim)
        self.edge_batch_norm = nn.BatchNorm1d(self.edge_mlp_out_dim)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Process node features
        node_features = x.float()
        node_features = x.float() 
        for layer in self.node_mlp_layers:
            node_features = layer(node_features)
            node_features = self.activation(node_features)
            node_features = self.dropout(node_features)

        # Apply Node BatchNorm after all node MLP layers
        node_features = self.node_batch_norm(node_features)

        # Process edge features
        edge_features = edge_attr.float() 
        for layer in self.edge_mlp_layers:
            edge_features = layer(edge_features)
            edge_features = self.activation(edge_features)
            edge_features = self.dropout(edge_features)

        # Apply Edge BatchNorm after all edge MLP layers
        edge_features = self.edge_batch_norm(edge_features)

        row, col = edge_index
        source_node_features = node_features[row]
        target_node_features = node_features[col]

        # Concatenate source node features, target node features, and edge features
        combined_features = torch.cat([source_node_features, target_node_features, edge_features], dim=1)

        switch_scores = torch.sigmoid(self.prediction_layer(combined_features))

        # Return a dictionary containing the prediction scores
        return {"switch_scores": switch_scores}
