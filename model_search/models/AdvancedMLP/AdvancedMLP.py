from enum import nonmember
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, SAGEConv, GraphConv
from typing import List, Optional, Dict, Any
from torch_geometric.nn import MessagePassing

def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    
    return activations[name.lower()]()


class GNNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, gnn_type: str = "GCN", 
                 heads: int = 4, dropout: float = 0.0, eps: float = 0.0):
        super().__init__()
        self.gnn_type = gnn_type.upper()
        self.output_dim = output_dim
        
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
        elif self.gnn_type == "SAGE":
            self.conv = SAGEConv(input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
    
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)
# class PhyR(nn.Module):
#     """Top-k straight-through selector; keeps gradient wrt probabilities."""
#     def __init__(self, k_ratio: float = 0.2):              # 20 % by default
#         super().__init__()
#         self.k_ratio = k_ratio

#     def forward(self, probs):                              # probs shape (E,)
#         k = max(1, int(self.k_ratio * probs.numel()))
#         idx = probs.topk(k, dim=0).indices
#         y_hard = torch.zeros_like(probs).scatter_(0, idx, 1.0)
#         return (y_hard - probs).detach() + probs  

class PhysicsTopK(nn.Module):
    def __init__(self, k_ratio: float):
        super().__init__()
        self.k_ratio = k_ratio

    def forward(self, probs: torch.Tensor, edge_batch: torch.Tensor):
        # probs: [E], edge_batch: [E] indicating graph idx for each edge
        num_graphs = int(edge_batch.max().item()) + 1
        counts = torch.bincount(edge_batch, minlength=num_graphs)  # [G]
        # compute k per graph (at least 1)
        k_per_graph = torch.clamp(torch.round(self.k_ratio * counts).long(), min=1)

        # build hard mask
        y_hard = torch.zeros_like(probs)
        for g in range(num_graphs):
            mask = (edge_batch == g).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                continue
            p = probs[mask]
            # per-graph topk (tie-breaker by index)
            k = min(k_per_graph[g].item(), p.numel())
            topk_vals, topk_idx = p.topk(k, largest=True, sorted=False)
            y_hard[mask[topk_idx]] = 1.0

        # straight‐through
        return (y_hard - probs).detach() + probs
             
class SwitchGatedMP(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_attr_dim, aggr="add"):
        super().__init__(aggr=aggr)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.res_lin = nn.Linear(in_channels, out_channels, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, out_channels),
            nn.Sigmoid()
        )
        self.act = nn.ReLU()

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.res_lin.weight)
        for m in self.gate_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr):
        # compute gate from edge features
        gate = self.gate_mlp(edge_attr)  # [E, out_channels]
        msg = self.lin(x_j)              # [E, out_channels]
        gated = msg * gate               # elementwise
        return self.act(gated + self.res_lin(x_i))

    def update(self, aggr_out):
        return self.act(aggr_out)

class MLPBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str = "relu",
                 dropout_rate: float = 0.1, use_batch_norm: bool = True, use_residual: bool = False):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_batch_norm else None
        self.use_residual = use_residual
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate)
        
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(dims[i + 1]))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            residual = x if (self.use_residual and x.shape[1] == layer.out_features) else None
            
            x = layer(x)
            if self.norms is not None:
                x = self.norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
            
            if residual is not None:
                x = x + residual
        
        return x


# ============================================================================
# Switch Prediction Heads
# ============================================================================

class MLPSwitchHead(nn.Module):
    """Traditional MLP-based switch prediction head."""
    
    def __init__(self, input_dim: int, hidden_layers: int = 3, activation: str = "relu",
                 dropout_rate: float = 0.1, use_batch_norm: bool = True):
        super().__init__()
        
        # Create decreasing layer sizes
        hidden_dims = [max(16, input_dim // (2 ** (i + 1))) for i in range(hidden_layers)]
        
        self.mlp = MLPBlock(
            input_dim, hidden_dims, activation, dropout_rate, use_batch_norm
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
        )
    
    def forward(self, x):
        x = self.mlp(x)
        return self.output(x)


class AttentionSwitchHead(nn.Module):
    """Self-attention based switch prediction head."""
    
    def __init__(self, input_dim: int, num_heads: int = 4, hidden_dim: int = 64,
                 activation: str = "relu", dropout_rate: float = 0.1):
        super().__init__()
        print(f"Embedding dim: {hidden_dim}, Num heads: {num_heads}")
        
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )
        
    def forward(self, x):
        h = self.input_projection(x).unsqueeze(1)  # [batch, 1, hidden]
        attended, _ = self.attention(h, h, h)
        h = self.layer_norm(h + attended)
        return self.output(h.squeeze(1))


class GraphAttentionSwitchHead(nn.Module):
    """Graph attention-based switch prediction head."""
    
    def __init__(self, node_dim: int, edge_dim: int, num_heads: int = 4,
                 hidden_dim: int = 64, activation: str = "relu", dropout_rate: float = 0.1):
        super().__init__()
        print(f"Embedding dim: {hidden_dim}, Num heads: {num_heads}")
        
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, source_nodes, target_nodes, edge_features, **kwargs):
        # Project to common dimension
        source_proj = self.node_proj(source_nodes)
        target_proj = self.node_proj(target_nodes)
        edge_proj = self.edge_proj(edge_features)
        
        # Stack as sequence: [source, edge, target]
        sequence = torch.stack([source_proj, edge_proj, target_proj], dim=1)
        attended, _ = self.attention(sequence, sequence, sequence)
        
        # Flatten and predict
        return self.output(attended.view(attended.size(0), -1))

class NodeVoltageGCN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = GraphConv(dim, dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, 1),
        )
    def forward(self, x,edge_index):
        h = self.conv(x, edge_index)
        return self.mlp(h).squeeze(1)
    
class EdgeFlowMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, 2),            # outputs [P̂, Q̂]
        )
    def forward(self, edge_feat):
        return self.net(edge_feat)
    
class AdvancedMLP(nn.Module):
    """
    Enhanced MLP with configurable GNN layers and sophisticated switch prediction heads.
    Now supports both binary and multiclass output modes.
    """
    
    def __init__(self, 
                 node_input_dim: int,
                 edge_input_dim: int,
                 # Output Configuration
                 output_type: str = "binary",  # "binary" or "multiclass"
                 num_classes: int = 2,
                 # GNN Configuration
                 gnn_type: Optional[str] = None,
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
                 use_skip_connections: bool = False,
                 pooling: str = "mean",
                 # Switch Head Configuration
                 switch_head_type: str = "mlp",
                 switch_head_layers: int = 3,
                 switch_attention_heads: int = 4,  
                 use_gated_mp: bool = False,
                 use_phyr: bool = False,
                 phyr_k_ratio: float = 0.2,
                 **kwargs):
        super().__init__()
        
        self.output_type = output_type.lower()
        self.num_classes = num_classes
        self.gnn_type = gnn_type
        self.use_node_mlp = use_node_mlp
        self.use_edge_mlp = use_edge_mlp
        self.use_skip_connections = use_skip_connections
        self.pooling = pooling
        self.switch_head_type = switch_head_type
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Validate output type
        if self.output_type not in ["binary", "multiclass"]:
            raise ValueError(f"output_type must be 'binary' or 'multiclass', got {self.output_type}")
        
        # ====================================================================
        # GNN Layers
        # ====================================================================
        if gnn_type and gnn_type.upper() != "NONE":
            self.gnn_layers = nn.ModuleList()
            current_dim = node_input_dim

            # first layer: optionally SwitchGatedMP
            if use_gated_mp:
                self.gnn_layers.append(
                    SwitchGatedMP(current_dim, gnn_hidden_dim, edge_input_dim)
                )
                current_dim = gnn_hidden_dim
            else:
                self.gnn_layers.append(
                    GNNLayer(current_dim, gnn_hidden_dim, gnn_type,
                             gat_heads, gat_dropout, gin_eps)
                )
                current_dim = gnn_hidden_dim

            # the remaining (gnn_layers-1) vanilla layers
            for _ in range(1, gnn_layers):
                self.gnn_layers.append(
                    GNNLayer(current_dim, gnn_hidden_dim, gnn_type,
                             gat_heads, gat_dropout, gin_eps)
                )
                current_dim = gnn_hidden_dim

            self.node_features_dim = current_dim
        else:
            self.gnn_layers = None
            self.node_features_dim = node_input_dim

        # store physics flags / modules
        self.use_gated_mp = use_gated_mp
        self.use_phyr = use_phyr                                            
        if use_phyr:
            self.phyr = PhysicsTopK(phyr_k_ratio)   
        self.switch_gate = None  
        
        # ====================================================================
        # MLP Processing
        # ====================================================================
        if use_node_mlp:
            self.node_mlp = MLPBlock(
                self.node_features_dim, node_hidden_dims, activation, 
                dropout_rate, use_batch_norm, use_residual
            )
            self.node_output_dim = node_hidden_dims[-1]
        else:
            self.node_mlp = None
            self.node_output_dim = self.node_features_dim
        
        if use_edge_mlp:
            self.edge_mlp = MLPBlock(
                edge_input_dim, edge_hidden_dims, activation, 
                dropout_rate, use_batch_norm, use_residual
            )
            self.edge_output_dim = edge_hidden_dims[-1]
        else:
            self.edge_mlp = None
            self.edge_output_dim = edge_input_dim
        
        # ====================================================================
        # Switch Prediction Head - Modified for Output Type
        # ====================================================================
        combined_dim = self.node_output_dim * 2 + self.edge_output_dim
        if use_skip_connections:
            combined_dim += node_input_dim * 2 + edge_input_dim
        
        # Determine output dimension based on output type
        if self.output_type == "multiclass":
            switch_output_dim = self.num_classes
        else:
            switch_output_dim = 1
        
        # Create switch head with appropriate output dimension
        if switch_head_type == "mlp":
            self.switch_head = self._create_mlp_head(
                combined_dim, switch_head_layers, activation, 
                dropout_rate, use_batch_norm, switch_output_dim
            )
            self.voltage_head = self._create_mlp_head(
                combined_dim, switch_head_layers, activation, 
                dropout_rate, use_batch_norm, 1  # Voltage is always single output
            )
        elif switch_head_type == "attention":
            self.switch_head = self._create_attention_head(
                combined_dim, switch_attention_heads, combined_dim, 
                activation, dropout_rate, switch_output_dim
            )
            self.voltage_head = self._create_attention_head(
                combined_dim, switch_attention_heads, combined_dim, 
                activation, dropout_rate, 1
            )
        elif switch_head_type == "graph_attention":
            self.switch_head = self._create_graph_attention_head(
                self.node_output_dim, self.edge_output_dim, switch_attention_heads, 
                combined_dim, activation, dropout_rate, switch_output_dim
            )
            self.voltage_head = self._create_graph_attention_head(
                self.node_output_dim, self.edge_output_dim, switch_attention_heads, 
                combined_dim, activation, dropout_rate, 1
            )
        else:
            raise ValueError(f"Unknown switch head type: {switch_head_type}")
        
        self.flow_head = EdgeFlowMLP(combined_dim)  # Use combined dim since we're passing combined_features
        self.voltage_head2 = NodeVoltageGCN(node_hidden_dims[-1] if use_node_mlp else self.node_features_dim)

    def _create_mlp_head(self, input_dim, layers, activation, dropout, use_bn, output_dim):
        """Create MLP head with specified output dimension."""
        hidden_dims = [max(16, input_dim // (2 ** (i + 1))) for i in range(layers)]
        mlp = MLPBlock(input_dim, hidden_dims, activation, dropout, use_bn)
        output_layer = nn.Linear(hidden_dims[-1], output_dim)
        return nn.Sequential(mlp, output_layer)
    
    def _create_attention_head(self, input_dim, num_heads, hidden_dim, activation, dropout, output_dim):
        """Create attention head with specified output dimension."""
        input_projection = nn.Linear(input_dim, hidden_dim)
        attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        layer_norm = nn.LayerNorm(hidden_dim)
        output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        class AttentionHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_projection = input_projection
                self.attention = attention
                self.layer_norm = layer_norm
                self.output = output_layers
                
            def forward(self, x):
                h = self.input_projection(x).unsqueeze(1)
                attended, _ = self.attention(h, h, h)
                h = self.layer_norm(h + attended)
                return self.output(h.squeeze(1))
        
        return AttentionHead()
    
    def _create_graph_attention_head(self, node_dim, edge_dim, num_heads, hidden_dim, 
                                   activation, dropout, output_dim):
        """Create graph attention head with specified output dimension."""
        node_proj = nn.Linear(node_dim, hidden_dim)
        edge_proj = nn.Linear(edge_dim, hidden_dim)
        attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        class GraphAttentionHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.node_proj = node_proj
                self.edge_proj = edge_proj
                self.attention = attention
                self.output = output_layers
                
            def forward(self, source_nodes, target_nodes, edge_features, **kwargs):
                source_proj = self.node_proj(source_nodes)
                target_proj = self.node_proj(target_nodes)
                edge_proj = self.edge_proj(edge_features)
                
                sequence = torch.stack([source_proj, edge_proj, target_proj], dim=1)
                attended, _ = self.attention(sequence, sequence, sequence)
                
                return self.output(attended.view(attended.size(0), -1))
        
        return GraphAttentionHead()
    
    def forward(self, data) -> Dict[str, torch.Tensor]:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', None)
        E = edge_index.size(1) 
       
        # Store original features for skip connections
        original_x = x.float()
        original_edge_attr = edge_attr.float()

        if self.use_gated_mp and self.switch_gate is None:                    
            self.switch_gate = nn.Parameter(torch.randn(E, 1).to(x.device)) 
        
        # ====================================================================
        # GNN Processing
        # ====================================================================
        node_features = x.float()
        for idx, gnn_layer in enumerate(self.gnn_layers or []):
            if self.use_gated_mp and idx == 0:                                
                node_features = gnn_layer(
                    node_features,
                    edge_index,
                    original_edge_attr    
                )
            else:
                node_features = gnn_layer(node_features, edge_index)
            node_features = self.activation(node_features)
            node_features = self.dropout(node_features)
        
        # ====================================================================
        # MLP Processing
        # ====================================================================
        if self.node_mlp is not None:
            node_features = self.node_mlp(node_features)
        
        edge_features = original_edge_attr
        if self.edge_mlp is not None:
            edge_features = self.edge_mlp(edge_features)
        
        # ====================================================================
        # Switch Prediction
        # ====================================================================
        row, col = edge_index
        source_node_features = node_features[row]
        target_node_features = node_features[col]
        
        # Always create combined features (needed for flow prediction too)
        combined_features = torch.cat([
            source_node_features, target_node_features, edge_features
        ], dim=1)
        
        # Add skip connections if enabled
        if self.use_skip_connections:
            skip_features = torch.cat([
                original_x[row], original_x[col], original_edge_attr
            ], dim=1)
            combined_features = torch.cat([combined_features, skip_features], dim=1)
        
        if self.switch_head_type == "graph_attention":
            switch_logits = self.switch_head(
                source_node_features, target_node_features, edge_features
            )
            voltage_logits = self.voltage_head(
                source_node_features, target_node_features, edge_features
            )
        else:
            # Use pre-computed combined features
            switch_logits = self.switch_head(combined_features)
            voltage_logits = self.voltage_head(combined_features)

        # Squeeze voltage logits (always single output)
        if voltage_logits.dim() > 1 and voltage_logits.size(-1) == 1:
            voltage_logits = voltage_logits.squeeze(-1)

        # Process switch outputs based on output type
        if self.output_type == "multiclass":
            # switch_logits shape: [E, num_classes]
            switch_probs_full = F.softmax(switch_logits, dim=1)
            # For downstream compatibility, extract "closed" probability (class 1)
            switch_probs = switch_probs_full[:, 1] if self.num_classes == 2 else switch_probs_full[:, -1]
        else:
            # Binary case: switch_logits shape: [E, 1] or [E]
            if switch_logits.dim() > 1 and switch_logits.size(-1) == 1:
                switch_logits = switch_logits.squeeze(-1)
            switch_probs = torch.sigmoid(switch_logits)
            switch_probs_full = None

        # Physics and flow predictions
        flows_hat = self.flow_head(combined_features)  # Use combined features for better flow prediction
        volt_hat = self.voltage_head2(node_features, edge_index) 

        outputs = {
            "switch_logits": switch_logits,          # Raw logits for loss computation
            "switch_predictions": switch_probs,      # Binary probabilities for physics/downstream
            "voltage_predictions": voltage_logits,
            "node_embeddings": node_features,
            "edge_embeddings": edge_features,
            "flows": flows_hat,                      # for physics_loss
            "node_v": volt_hat,                      # for physics_loss
        }

        # Add multiclass probabilities if available
        if switch_probs_full is not None:
            outputs["switch_probabilities"] = switch_probs_full
        
        # Apply PhysicsTopK if enabled
        if self.use_phyr:
            if batch is not None:
                # Get edge batch indices
                edge_batch = batch[edge_index[0]]  # Use source node batch indices
            else:
                # Single graph case
                edge_batch = torch.zeros(E, dtype=torch.long, device=x.device)
            
            switch_mask = self.phyr(switch_probs, edge_batch)  
            outputs["switch_predictions"] = switch_mask  
            outputs["switch_mask"] = switch_mask   
        
        # Add gated message passing features if available
        if self.use_gated_mp and self.switch_gate is not None:
            outputs["switch_gates"] = torch.sigmoid(self.switch_gate).squeeze(1)
        
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model configuration and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "output_type": self.output_type,
            "num_classes": self.num_classes,
            "gnn_type": self.gnn_type,
            "switch_head_type": self.switch_head_type,
            "use_node_mlp": self.use_node_mlp,
            "use_edge_mlp": self.use_edge_mlp,
            "use_skip_connections": self.use_skip_connections,
            "pooling": self.pooling,
        }

# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable
    }


def initialize_weights(model: nn.Module, init_type: str = "xavier"):
    """Initialize model weights."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if init_type == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)