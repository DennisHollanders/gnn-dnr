import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16],
                 latent_dim=8, activation='prelu', dropout_rate=0.0):
        super(GraphAutoencoder, self).__init__()
        activation_map = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'selu': F.selu,
            'prelu': F.prelu,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh
        }
        self.activation = activation_map.get(activation, F.relu)
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        layer_sizes = [input_dim] + hidden_dims
        for i in range(len(layer_sizes)-1):
            self.encoder_layers.append(pyg_nn.GCNConv(layer_sizes[i], layer_sizes[i+1]))
        self.fc_latent = nn.Linear(hidden_dims[-1], latent_dim)
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        reversed_dims = list(reversed(hidden_dims))
        layer_sizes = [latent_dim] + reversed_dims
        for i in range(len(layer_sizes)-1):
            self.decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.decoder_out = nn.Linear(reversed_dims[-1], input_dim)
        # BatchNorms and dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_batchnorms = nn.ModuleList([nn.BatchNorm1d(dim) for dim in hidden_dims])
        self.decoder_batchnorms = nn.ModuleList([nn.BatchNorm1d(dim) for dim in reversed_dims])
        
    def encode(self, data):
        x, edge_index = data.x.float(), data.edge_index
        for layer, bn in zip(self.encoder_layers, self.encoder_batchnorms):
            x = layer(x, edge_index)
            x = self.activation(x)
            x = bn(x)
        latent = self.fc_latent(x)
        return latent

    def decode(self, z):
        for layer, bn in zip(self.decoder_layers, self.decoder_batchnorms):
            z = layer(z)
            z = self.activation(z)
            z = bn(z)
        return self.decoder_out(z)
        
    def forward(self, data):
        z = self.encode(data)
        out = self.decode(z)
        return out
