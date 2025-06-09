import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_batch
from typing import List, Optional, Dict, Any
from torch_geometric.nn import MessagePassing






class AdvancedMLP(nn.Module):
    """
    Advanced MLP with configurable GNN layers, multiple pooling strategies,
    and sophisticated architecture options for hyperparameter optimization.
    """
    
    def __init__(self, 
                 node_input_dim: int,
                 edge_input_dim: int,
                 # GNN Configuration
                 gnn_type: Optional[str] = None,  # "GCN", "GAT", "GIN", or None
                 gnn_layers: int = 2,
                 gnn_hidden_dim: int = 64,
                 gat_heads: int = 4,
                 gat_dropout: float = 0.1,
                 gin_eps: float = 0.0,
                 # MLP Configuration
                 use_node_mlp: bool = True,
                 use_edge_mlp: bool = True,
                 node_hidden_dims: List[int] = [64, 32],
                 edge_hidden_dims: List[int] = [64, 32],
                 # General Configuration
                 activation: str = "relu",
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True,
                 use_residual: bool = False,
                 pooling: str = "mean",  # "mean", "max", "add", "attention"
                 **kwargs):
        super().__init__()
        
        self.gnn_type = gnn_type
        self.use_node_mlp = use_node_mlp
        self.use_edge_mlp = use_edge_mlp
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.pooling = pooling
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate)
        
        # ====================================================================
        # GNN Layers 
        # ====================================================================
        if gnn_type and gnn_type.upper() != "NONE":
            self.gnn_layers = nn.ModuleList()
            current_dim = node_input_dim
            
            for i in range(gnn_layers):
                output_dim = gnn_hidden_dim if i < gnn_layers - 1 else gnn_hidden_dim
                
                gnn_layer = GNNLayer(
                    input_dim=current_dim,
                    output_dim=output_dim,
                    gnn_type=gnn_type,
                    heads=gat_heads,
                    dropout=gat_dropout,
                    eps=gin_eps
                )
                self.gnn_layers.append(gnn_layer)
                current_dim = output_dim
            
            self.node_features_dim = current_dim
        else:
            self.gnn_layers = None
            self.node_features_dim = node_input_dim
        
        # ====================================================================
        # Node MLP 
        # ====================================================================
        if use_node_mlp:
            self.node_mlp_layers = nn.ModuleList()
            self.node_batch_norms = nn.ModuleList() if use_batch_norm else None
            
            in_dim = self.node_features_dim
            for i, out_dim in enumerate(node_hidden_dims):
                self.node_mlp_layers.append(nn.Linear(in_dim, out_dim))
                
                if use_batch_norm:
                    self.node_batch_norms.append(nn.BatchNorm1d(out_dim))
                
                # Add residual blocks if specified and dimensions match
                if use_residual and in_dim == out_dim:
                    setattr(self, f'node_residual_{i}', 
                           ResidualBlock(out_dim, activation, dropout_rate))
                
                in_dim = out_dim
            
            self.node_output_dim = in_dim
        else:
            self.node_mlp_layers = None
            self.node_output_dim = self.node_features_dim
        
        # ====================================================================
        # Edge MLP 
        # ====================================================================
        if use_edge_mlp:
            self.edge_mlp_layers = nn.ModuleList()
            self.edge_batch_norms = nn.ModuleList() if use_batch_norm else None
            
            in_dim = edge_input_dim
            for i, out_dim in enumerate(edge_hidden_dims):
                self.edge_mlp_layers.append(nn.Linear(in_dim, out_dim))
                
                if use_batch_norm:
                    self.edge_batch_norms.append(nn.BatchNorm1d(out_dim))
                
                # Add residual blocks if specified and dimensions match
                if use_residual and in_dim == out_dim:
                    setattr(self, f'edge_residual_{i}', 
                           ResidualBlock(out_dim, activation, dropout_rate))
                
                in_dim = out_dim
            
            self.edge_output_dim = in_dim
        else:
            self.edge_mlp_layers = None
            self.edge_output_dim = edge_input_dim
        
        # ====================================================================
        # Pooling Layer
        # ====================================================================
        if pooling == "attention":
            self.attention_pool = AttentionPooling(self.node_output_dim)
        else:
            self.attention_pool = None
        
        # ====================================================================
        # Final Prediction Layer
        # ====================================================================
        # Combine node features from both endpoints of edge + edge features
        combined_dim = self.node_output_dim * 2 + self.edge_output_dim
        
        self.prediction_layers = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            self.activation,
            self.dropout,
            nn.Linear(combined_dim // 2, combined_dim // 4),
            self.activation,
            self.dropout,
            nn.Linear(combined_dim // 4, 1),
            nn.Sigmoid()
        )
        
    
        if use_batch_norm:
            self.prediction_layers.insert(1, nn.BatchNorm1d(combined_dim // 2))
            self.prediction_layers.insert(5, nn.BatchNorm1d(combined_dim // 4))
    
    def forward(self, data) -> Dict[str, torch.Tensor]:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', None)
        
        # ====================================================================
        # Process Node Features
        # ====================================================================
        node_features = x.float()
        
        if self.gnn_layers is not None:
            for gnn_layer in self.gnn_layers:
                node_features = gnn_layer(node_features, edge_index)
                node_features = self.activation(node_features)
                node_features = self.dropout(node_features)
        
        if self.node_mlp_layers is not None:
            for i, layer in enumerate(self.node_mlp_layers):
                residual = node_features if (self.use_residual and 
                                           hasattr(self, f'node_residual_{i}') and 
                                           node_features.shape[1] == layer.out_features) else None
                
                node_features = layer(node_features)
                
                if self.use_batch_norm and self.node_batch_norms:
                    node_features = self.node_batch_norms[i](node_features)
                
                node_features = self.activation(node_features)
                node_features = self.dropout(node_features)
                
                if residual is not None:
                    residual_block = getattr(self, f'node_residual_{i}')
                    node_features = residual_block(node_features)
        
        # ====================================================================
        # Process Edge Features
        # ====================================================================
        edge_features = edge_attr.float()
        
        if self.edge_mlp_layers is not None:
            for i, layer in enumerate(self.edge_mlp_layers):
                residual = edge_features if (self.use_residual and 
                                           hasattr(self, f'edge_residual_{i}') and
                                           edge_features.shape[1] == layer.out_features) else None
                
                edge_features = layer(edge_features)
                
                if self.use_batch_norm and self.edge_batch_norms:
                    edge_features = self.edge_batch_norms[i](edge_features)
                
                edge_features = self.activation(edge_features)
                edge_features = self.dropout(edge_features)
                
                if residual is not None:
                    residual_block = getattr(self, f'edge_residual_{i}')
                    edge_features = residual_block(edge_features)
        
        # ====================================================================
        # Combine Features for Edge Prediction
        # ====================================================================
        row, col = edge_index
        source_node_features = node_features[row]
        target_node_features = node_features[col]
        
        combined_features = torch.cat([
            source_node_features, 
            target_node_features, 
            edge_features
        ], dim=1)
        
        switch_scores = self.prediction_layers(combined_features)
        
        outputs = {
            "switch_scores": switch_scores,
            "node_embeddings": node_features,
            "edge_embeddings": edge_features,
            "combined_features": combined_features
        }
        
        # Add graph-level representations if batch is available
        if batch is not None:
            if self.pooling == "mean":
                graph_repr = global_mean_pool(node_features, batch)
            elif self.pooling == "max":
                graph_repr = global_max_pool(node_features, batch)
            elif self.pooling == "add":
                graph_repr = global_add_pool(node_features, batch)
            elif self.pooling == "attention" and self.attention_pool:
                graph_repr = self.attention_pool(node_features, batch)
            else:
                graph_repr = global_mean_pool(node_features, batch)
            
            outputs["graph_representation"] = graph_repr
        
        return outputs

def get_activation(name: str) -> nn.Module:
    """Helper function to get activation layer by name."""
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "prelu": nn.PReLU(),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "swish": nn.SiLU(),
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation function: {name}. Available: {list(activations.keys())}")
    
    return activations[name.lower()]


class ResidualBlock(nn.Module):
    """Residual block for MLP layers"""
    
    def __init__(self, dim: int, activation: str = "relu", dropout_rate: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.activation(x + residual)


class AttentionPooling(nn.Module):
    """Attention-based pooling for graph-level representations"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
        
    def forward(self, x, batch):
        # Compute attention weights
        attention_weights = self.attention(x)  # [num_nodes, 1]
        attention_weights = torch.softmax(attention_weights, dim=0)
        
        # Apply attention weights and pool
        weighted_x = x * attention_weights
        return global_add_pool(weighted_x, batch)


class GNNLayer(nn.Module):
    """Configurable GNN layer supporting different architectures"""
    
    def __init__(self, input_dim: int, output_dim: int, gnn_type: str = "GCN", 
                 heads: int = 4, dropout: float = 0.0, eps: float = 0.0):
        super().__init__()
        self.gnn_type = gnn_type.upper()
        
        if self.gnn_type == "GCN":
            self.conv = GCNConv(input_dim, output_dim)
        elif self.gnn_type == "GAT":
            self.conv = GATv2Conv(input_dim, output_dim, heads=heads, 
                                dropout=dropout, concat=False)
        elif self.gnn_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
            self.conv = GINConv(mlp, eps=eps)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
    
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model configuration and parameter count"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "gnn_type": self.gnn_type,
            "use_node_mlp": self.use_node_mlp,
            "use_edge_mlp": self.use_edge_mlp,
            "use_batch_norm": self.use_batch_norm,
            "use_residual": self.use_residual,
            "pooling": self.pooling,
        }
        
        return info


# ============================================================================
# Custom Loss Functions
# ============================================================================

class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for imbalanced switch prediction"""
    
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    
    def forward(self, predictions, targets):
        return self.bce(predictions.squeeze(), targets.float())


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        predictions = predictions.squeeze()
        targets = targets.float()
        
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss with switch prediction and physics constraints"""
    
    def __init__(self, weight_switch: float = 1.0, weight_physics: float = 0.1):
        super().__init__()
        self.weight_switch = weight_switch
        self.weight_physics = weight_physics
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions, targets):
        # Main switch prediction loss
        switch_loss = self.bce_loss(predictions.squeeze(), targets.float())
        
        # Physics constraint: encourage sparse switching
        sparsity_loss = torch.mean(predictions)
        
        total_loss = (self.weight_switch * switch_loss + 
                     self.weight_physics * sparsity_loss)
        
        return total_loss


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable
    }


def initialize_weights(model: nn.Module, init_type: str = "xavier"):
    """Initialize model weights"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if init_type == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_type == "normal":
                nn.init.normal_(module.weight, 0, 0.01)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


