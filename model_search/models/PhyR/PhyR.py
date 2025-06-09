import torch
import torch.nn as nn
from torch.autograd import Function
from torch_geometric.nn import MessagePassing


class PFMessagePassingLayer(MessagePassing):
    def __init__(self, in_feats, out_feats):
        super(PFMessagePassingLayer, self).__init__(aggr='add')  # sum aggregation
        self.lin_src = nn.Linear(in_feats, out_feats)
        self.lin_dst = nn.Linear(in_feats, out_feats)
        self.edge_lin = nn.Linear(1, out_feats)  # for admittance magnitude

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, in_feats], edge_attr: [num_edges, 1] (|Y_ij|)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: features of neighbor j
        m_j = self.lin_src(x_j)
        e_ij = self.edge_lin(edge_attr)
        return m_j * e_ij

    def update(self, aggr_out, x):
        # Combine aggregated message with self node representation
        return torch.relu(self.lin_dst(x) + aggr_out)

class EdgeNetwork(nn.Module):
    def __init__(self, edge_in_feats, hidden_feats, out_feats):
        super(EdgeNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_in_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )
    def forward(self, e):
        # e: [num_edges, edge_in_feats]
        return self.mlp(e)  # [num_edges, out_feats]
    
class PFMessagePassingMultiHead(MessagePassing):
    def __init__(self, in_feats, out_feats, edge_in_feats, hidden_edge_feats, heads=4):
        super(PFMessagePassingMultiHead, self).__init__(aggr='add')
        assert out_feats % heads == 0, "out_feats must be divisible by heads"
        self.heads = heads
        self.head_dim = out_feats // heads

        # Node linear for each head (shared or separate?)
        self.lin_node = nn.ModuleList([
            nn.Linear(in_feats, self.head_dim, bias=False) for _ in range(heads)
        ])
        # Per-head attention vector a_h (dimension 3*head_dim if concatenating [W x_i || W x_j || phi(e)])
        self.attn = nn.ModuleList([
            nn.Linear(3*self.head_dim, 1, bias=False) for _ in range(heads)
        ])
        # Edge MLP (option 1: share across heads; option 2: separate)
        self.edge_network = EdgeNetwork(edge_in_feats, hidden_edge_feats, self.head_dim)

        # Final linear to mix heads output back to out_feats
        self.final_lin = nn.Linear(out_feats, out_feats)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr):
        """
        x: [num_nodes, in_feats]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_in_feats]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        """
        PyG will automatically pass x_i = x[edge_index[0]], x_j = x[edge_index[1]]
        index: [num_edges] indicates target node indices
        """
        # For each head, compute attention and message
        all_head_msgs = []
        for h in range(self.heads):
            # 1) Linear projections
            Wh_i = self.lin_node[h](x_i)  # [num_edges, head_dim]
            Wh_j = self.lin_node[h](x_j)  # [num_edges, head_dim]

            # 2) Edge embedding via shared EdgeNetwork
            E_h = self.edge_network(edge_attr)  # [num_edges, head_dim]

            # 3) Compute attention logits: a_h^T [ Wh_i || Wh_j || E_h ]
            cat_feat   = torch.cat([Wh_i, Wh_j, E_h], dim=-1)  # [num_edges, 3*head_dim]
            e_h_ij     = self.leaky_relu(self.attn[h](cat_feat))  # [num_edges, 1]

            # 4) Softmax over neighbors for each target node
            alpha_h = torch.softmax(e_h_ij, index)  # [num_edges, 1] (PyG’s softmax normalizes per index)
            
            # 5) Message: multiply neighbor embedding by attention and edge features
            m_h = alpha_h * (Wh_j + E_h)  # [num_edges, head_dim]

            all_head_msgs.append(m_h)

        # 6) Concatenate heads along feature dim: [num_edges, heads*head_dim]
        multi_head_msg = torch.cat(all_head_msgs, dim=-1)  # [num_edges, out_feats]
        return multi_head_msg

    def update(self, aggr_out, x):
        """
        aggr_out: [num_nodes, out_feats] after summing messages per node
        x:         [num_nodes, in_feats]
        """
        # We could optionally transform x via a linear into out_feats to add residual
        x_res = torch.cat([self.lin_node[h](x) for h in range(self.heads)], dim=-1)  # [num_nodes, out_feats]
        # Node update MLP (with residual)
        updated = torch.relu(self.final_lin(aggr_out + x_res))  # [num_nodes, out_feats]
        return updated


class GPhyR(Function):
    @staticmethod
    def forward(ctx, s, edge_index, num_nodes):
        """
        s: [num_edges] continuous switch scores in (0,1)
        edge_index: [2, num_switch_edges] listing the endpoints of each switchable edge
        num_nodes: integer, N
        Returns a binary vector b in {0,1}^num_edges where exactly N-1 entries are 1
        and they form a spanning tree. We use a greedy STOUT (soft‐MST) in forward.
        """
        num_edges = s.shape[0]
        # 1) Add Gumbel noise to s to break ties in ranking
        gumbel = -torch.log(-torch.log(torch.rand_like(s) + 1e-9) + 1e-9)
        scores = s + gumbel  # [num_edges]

        # 2) Sort edges by descending scores
        sorted_indices = torch.argsort(scores, descending=True)

        # 3) Kruskal‐like MST: union-find to pick edges
        parent = list(range(num_nodes))
        rank   = [0]*num_nodes
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        def union(u, v):
            ru, rv = find(u), find(v)
            if ru == rv: return False
            # union by rank
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            return True

        b = torch.zeros(num_edges, device=s.device)
        edges_used = 0
        for idx in sorted_indices.tolist():
            i, j = edge_index[0, idx].item(), edge_index[1, idx].item()
            if union(i, j):
                b[idx] = 1.0
                edges_used += 1
                if edges_used == num_nodes - 1:
                    break

        ctx.save_for_backward(s)  # we’ll need s to pass gradients
        return b  # binary tensor [num_edges]

    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight‐through estimator: pass gradient from b to s directly.
        grad_output: [num_edges], gradient wrt b
        We simply return grad_output for s, None for others.
        """
        (s,) = ctx.saved_tensors
        return grad_output.clone(), None, None

# Usage in a nn.Module:
class GPhyRLayer(nn.Module):
    def __init__(self):
        super(GPhyRLayer, self).__init__()

    def forward(self, s, edge_index, num_nodes):
        """
        s: [num_edges], continuous logits from preceding GNN (e.g. via sigmoid).
        edge_index: [2, num_edges]
        num_nodes: integer N
        """
        # 1) Apply sigmoid to logits to get s in (0,1)
        s_prob = torch.sigmoid(s)
        # 2) Round via differentiable MST
        b = GPhyR.apply(s_prob, edge_index, num_nodes)  
        # b: [num_edges] binary vector with exactly N-1 edges = 1
        return b
    
class CycleGatingPFLayer(MessagePassing):
    def __init__(self, in_feats, out_feats, edge_in_feats, hidden_edge_feats, heads=4):
        super(CycleGatingPFLayer, self).__init__(aggr='add')
        # ... (same as PFMessagePassingEnhanced: define edge_network, lin_node, attn, etc.)
        # Add a cycle penalty (learnable scalar)
        self.log_gamma = nn.Parameter(torch.tensor(0.0))  # log of cycle penalty

        # We also need to track parent pointers in training/inference
        # We will store them as metadata per node during forward pass
        # Let parent_pointers: [num_nodes], each entry is the parent node index or -1 if root
        # Cycle detection: find(u) and find(v)
        # We’ll maintain a union-find structure in CPU or GPU

    def forward(self, x, edge_index, edge_attr, parent_pointers):
        """
        parent_pointers: [num_nodes] tensor with parent pointer for each node (-1 for no parent)
        (In practice, we can keep a union-find per batch to track connectivity.)
        """
        self.parent_pointers = parent_pointers.clone()
        return self.propagate(edge_index, x=x, edge_attr=edge_attr,
                              parent_pointers=parent_pointers)

    def message(self, x_i, x_j, edge_attr, index, parent_pointers):
        """
        We have x_i, x_j (node embeddings), edge_attr, index, and parent_pointers.
        """
        all_heads = []
        gamma = torch.sigmoid(self.log_gamma)  # cycle penalty in (0,1)

        # Precompute edge embeddings
        E_shared = self.edge_network(edge_attr)  # [num_edges, head_dim]

        # For each edge (j -> i), check if it creates a cycle:
        # If parent_pointers[i] != -1 and parent_pointers[j] != -1 and find(i) == find(j) 
        # then this edge forms a cycle under the current parent structure.
        # We'll simulate Union-Find find() calls in a batched manner (pseudo).
        # For simplicity, we can precompute a boolean mask "forms_cycle_mask"
        # of shape [num_edges], where True indicates edge would form cycle.

        forms_cycle_mask = self._detect_cycles(edge_index, parent_pointers)  # [num_edges], bool

        # Now proceed to compute attention‐based messages per head
        for h in range(self.heads):
            Wh_i = self.lin_node[h](x_i)  # [num_edges, head_dim]
            Wh_j = self.lin_node[h](x_j)  # [num_edges, head_dim]
            E_h  = E_shared

            cat_feat = torch.cat([Wh_i, Wh_j, E_h], dim=-1)  # [num_edges, 3*head_dim]
            e_h_ij   = self.leaky_relu(self.attn[h](cat_feat))  # [num_edges, 1]
            alpha_h  = torch.softmax(e_h_ij, index)  # [num_edges, 1]

            m_h = alpha_h * (Wh_j + E_h)  # [num_edges, head_dim]

            # Apply cycle penalty: if form cycle, multiply by gamma
            cycle_mask = forms_cycle_mask.unsqueeze(-1).float()  # [num_edges,1]
            penalty  = 1.0 - cycle_mask + cycle_mask * gamma  # if cycle_mask=1 => gamma, else 1
            m_h = m_h * penalty  # attenuate cycle-forming messages

            all_heads.append(m_h)

        multi_head_msg = torch.cat(all_heads, dim=-1)
        return multi_head_msg

    def update(self, aggr_out, x):
        # Similar to PFMessagePassingEnhanced's update
        x_res = torch.cat([self.lin_node[h](x) for h in range(self.heads)], dim=-1)
        y = self.final_lin(aggr_out + x_res)
        y = self.node_update_mlp(y)
        return torch.relu(y)

    def _detect_cycles(self, edge_index, parent_pointers):
        """
        edge_index: [2, num_edges]
        parent_pointers: [num_nodes] with parent for each node
        Returns a boolean tensor [num_edges] = True if edge (j->i) would form a cycle.
        """
        # Very simplified: We check if i and j already share the same root in parent_pointers.
        # In practice, maintain a union-find for current parents.
        num_edges = edge_index.shape[1]
        forms_cycle = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)

        # Build union-find from parent pointers
        parent = parent_pointers.clone().tolist()  # [num_nodes]
        def find(u):
            if parent[u] == -1:
                return u
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        # For each edge (j -> i)
        for idx in range(num_edges):
            i = edge_index[0, idx].item()
            j = edge_index[1, idx].item()
            ri = find(i); rj = find(j)
            if ri == rj:
                forms_cycle[idx] = True
        return forms_cycle
class PFMessagePassingEnhanced(MessagePassing):
    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 edge_in_feats: int,
                 hidden_edge_feats: int,
                 heads: int = 4,
                 node_hidden_feats: int = 128,
                 dropout: float = 0.0):
        """
        - in_feats:  number of node input features.
        - out_feats: number of node output features (must be divisible by heads).
        - edge_in_feats: number of raw edge features (|Y|, angle, etc.).
        - hidden_edge_feats: size of hidden layers in edge MLP.
        - heads: number of multi-heads (out_feats % heads == 0).
        - node_hidden_feats: hidden dimension in node-update MLP.
        - dropout: dropout probability on final outputs.
        """
        super(PFMessagePassingEnhanced, self).__init__(aggr='add')
        assert out_feats % heads == 0, "out_feats must be divisible by heads"
        self.heads      = heads
        self.head_dim   = out_feats // heads
        self.dropout    = nn.Dropout(dropout)

        # 1) Edge MLP (shared by heads)
        self.edge_network = EdgeNetwork(edge_in_feats, hidden_edge_feats, self.head_dim)

        # 2) Node projections for each head
        self.lin_node = nn.ModuleList([
            nn.Linear(in_feats, self.head_dim, bias=False) for _ in range(heads)
        ])
        # 3) Per-head attention vectors
        self.attn = nn.ModuleList([
            nn.Linear(3*self.head_dim, 1, bias=False) for _ in range(heads)
        ])
        self.leaky_relu = nn.LeakyReLU(0.2)

        # 4) Final mixing linear (post-aggregation)
        self.final_lin = nn.Linear(out_feats, out_feats)

        # 5) Node-update MLP
        self.node_update_mlp = nn.Sequential(
            nn.Linear(out_feats, node_hidden_feats),
            nn.ReLU(),
            nn.LayerNorm(node_hidden_feats),
            nn.Linear(node_hidden_feats, out_feats)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x:         [num_nodes, in_feats]
        edge_index:[2, num_edges]
        edge_attr: [num_edges, edge_in_feats]
        """
        # (PyG will call "message" and "update" behind the scenes)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index):
        """
        x_i: [num_edges, in_feats] features of target nodes (for each edge).
        x_j: [num_edges, in_feats] features of source neighbors.
        edge_attr: [num_edges, edge_in_feats]
        index: target node indices for each edge (used for softmax).
        """
        all_heads = []
        # 1) Precompute edge embeddings once: [num_edges, head_dim]
        E_shared = self.edge_network(edge_attr)

        for h in range(self.heads):
            Wh_i = self.lin_node[h](x_i)    # [num_edges, head_dim]
            Wh_j = self.lin_node[h](x_j)    # [num_edges, head_dim]
            E_h  = E_shared                 # share across heads (or clone if separate)

            # 2) Attention logits
            cat_feat = torch.cat([Wh_i, Wh_j, E_h], dim=-1)  # [num_edges, 3*head_dim]
            e_h_ij   = self.leaky_relu(self.attn[h](cat_feat))  # [num_edges, 1]

            # 3) Normalize over neighbors: [num_edges, 1]
            alpha_h = torch.softmax(e_h_ij, index)

            # 4) Weighted message: [num_edges, head_dim]
            m_h = alpha_h * (Wh_j + E_h)

            all_heads.append(m_h)

        # 5) Concatenate all heads: [num_edges, out_feats]
        multi_head_msg = torch.cat(all_heads, dim=-1)
        return multi_head_msg

    def update(self, aggr_out, x):
        """
        aggr_out: [num_nodes, out_feats] after summing messages from neighbors.
        x:        [num_nodes, in_feats]
        """
        # 1) Residual path: project x into out_feats
        x_res = torch.cat([self.lin_node[h](x) for h in range(self.heads)], dim=-1)
        # 2) Sum + final linear
        y = self.final_lin(aggr_out + x_res)  # [num_nodes, out_feats]
        # 3) Node-update MLP + nonlinearity
        y = self.node_update_mlp(y)          # [num_nodes, out_feats]
        y = torch.relu(y)                    # [num_nodes, out_feats]
        # 4) Dropout if desired
        return self.dropout(y)

# (Assume PFMessagePassingEnhanced & PhyRLayer are defined as above)

class Phyr(nn.Module):
    def __init__(self,
                 num_layers: int,
                 node_in_feats: int,
                 edge_in_feats: int,
                 hidden_feats: int,
                 heads: int = 4):
        super(Phyr, self).__init__()
        self.num_layers = num_layers
        self.hidden_feats = hidden_feats
        self.heads = heads

        # 1) Initial node embedding (if needed)
        self.node_encoder = nn.Linear(node_in_feats, hidden_feats)

        # 2) Stack of enhanced PFMessagePassing layers
        self.layers = nn.ModuleList()
        for t in range(num_layers):
            self.layers.append(PFMessagePassingEnhanced(
                in_feats=hidden_feats,
                out_feats=hidden_feats,
                edge_in_feats=edge_in_feats,
                hidden_edge_feats=hidden_feats,
                heads=heads,
                node_hidden_feats=hidden_feats
            ))

        # 3) Readout heads
        self.voltage_head = nn.Linear(hidden_feats, 2)   # [V_mag, V_ang]
        self.switch_head  = nn.Sequential(
            nn.Linear(2*hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, 1)  # logit for each edge
        )

        # 4) PhyR layer for radiality enforcement
        self.phyr = GPhyRLayer()

    def forward(self, x, edge_index, edge_attr, switch_edge_mask):
        """
        x: [num_nodes, node_in_feats]
        edge_index: [2, num_edges]  # includes both non-switch & switch edges
        edge_attr: [num_edges, edge_in_feats]
        switch_edge_mask: [num_edges] boolean indicating which edges are switchable
        """
        # 1) Encode nodes
        h = self.node_encoder(x)  # [num_nodes, hidden_feats]

        # 2) Message passing (we do NOT update edge_attr here for simplicity)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        # 3) Voltage prediction per node
        V_pred = self.voltage_head(h)  # [num_nodes, 2]

        # 4) Switch logits per switchable edge
        src = edge_index[0]  # [num_edges]
        dst = edge_index[1]
        h_src = h[src]  # [num_edges, hidden_feats]
        h_dst = h[dst]

        # For non-switch edges, we do not need a switch logit; set to large neg value
        s_logits = torch.full((edge_index.shape[1], 1),
                              float('-1e9'), device=h.device)

        # Only compute logits where switch_edge_mask == True
        sw_indices = torch.nonzero(switch_edge_mask, as_tuple=False).view(-1)  # [num_switches]
        h_cat = torch.cat([h_src[sw_indices], h_dst[sw_indices]], dim=-1)  # [num_switches, 2*hidden_feats]
        s_logits_sw = self.switch_head(h_cat)  # [num_switches, 1]
        s_logits[sw_indices] = s_logits_sw

        # 5) Physics−Informed Rounding (PhyR) for switch decisions
        # get continuous scores in (0,1)
        s_scores = torch.sigmoid(s_logits[sw_indices].view(-1))  # [num_switches]
        b_sw = self.phyr(s_scores, edge_index[:, sw_indices], num_nodes=h.shape[0])
        # b_sw: [num_switches] binary, exactly N-1 edges chosen

        # 6) Assemble final adjacency: combine non-switch edges (always closed) + chosen switches
        final_switch_mask = torch.zeros_like(switch_edge_mask, dtype=torch.bool)
        final_switch_mask[sw_indices] = (b_sw > 0.5)  # boolean
        closed_edge_mask = (~switch_edge_mask) | final_switch_mask  # non-switch edges + chosen switches

        # closed_edge_mask: [num_edges] boolean indicating which edges are in final tree

        return V_pred, closed_edge_mask