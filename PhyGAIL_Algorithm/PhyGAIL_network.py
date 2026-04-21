import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

from PhyGAIL_Algorithm.PhyGAIL_config import *

# ==============================
# Physics-Inspired Gated Message Passing Layer
# ==============================
class PhysicsGatedMessagePassing(MessagePassing):
    """Message-passing layer with learned attraction and repulsion gates."""
    def __init__(self, node_dim, hidden_dim, dropout=0.2):
        super().__init__(aggr='add') 
        
        self.edge_type_embedding = nn.Embedding(3, hidden_dim)
        input_dim = node_dim * 2 + hidden_dim
        
        self.msg_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.strength_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.node_updater = nn.Sequential(
            nn.LayerNorm(node_dim + hidden_dim),
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(node_dim, node_dim)

    def forward(self, x, edge_index, edge_type):
        out = self.propagate(edge_index, x=x, edge_type=edge_type)
        
        # Node update with residual connection
        out = torch.cat([x, out], dim=-1)
        out = self.node_updater(out)
        out = F.relu(out + self.residual(x))
        return self.dropout(out)

    def message(self, x_i, x_j, edge_type):
        relative_x = x_j - x_i  
        type_emb = self.edge_type_embedding(edge_type)
        raw_input = torch.cat([x_i, relative_x, type_emb], dim=-1)
        edge_features = self.msg_mlp(raw_input)
        
        gates = self.gate_network(edge_features) 
        gates = torch.sigmoid(gates)
        
        attract_w = gates[:, 0:1]
        repel_w = gates[:, 1:2]
        
        is_center = (edge_type == 2).unsqueeze(1).float()
        attract_w = torch.clamp(attract_w + is_center * 0.5, 0.0, 1.0) 
        
        strength = self.strength_network(edge_features)
        strength = F.softplus(strength)
        
        weighted_message = edge_features * strength * (attract_w - repel_w)
        
        return weighted_message


# ==============================
# Physics-Inspired Gated GNN Encoder
# ==============================
class PhysicsInformedGNNEncoder(nn.Module):
    """Physics-informed encoder used by the PhyGAIL policy and value networks."""
    
    def __init__(self, node_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Node type encoding
        self.type_embedding = nn.Embedding(3, hidden_dim // 4)
        
        # Initial encoding layer
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim + hidden_dim // 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Gated message passing layers
        self.gated_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = PhysicsGatedMessagePassing(
                hidden_dim, hidden_dim, dropout
            )
            self.gated_layers.append(layer)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        """Encode a graph batch into node embeddings."""
        type_emb = self.type_embedding(data.node_type)
        x = torch.cat([data.x, type_emb], dim=-1)
        x = self.node_encoder(x)
        
        for layer in self.gated_layers:
            x = layer(x, data.edge_index, data.edge_type)
            x = self.dropout(x)
        
        return x
class _StandardGraphEncoder(nn.Module):
    """Shared implementation for non-physics ablation encoders."""

    def __init__(self, node_dim, hidden_dim, num_layers=2, dropout=0.1, **conv_kwargs):
        super().__init__()
        self.type_embedding = nn.Embedding(3, hidden_dim // 4)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim + hidden_dim // 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        self.convs = nn.ModuleList(self._build_convs(hidden_dim, num_layers, dropout, **conv_kwargs))
        self.dropout = nn.Dropout(dropout)

    def _build_convs(self, hidden_dim, num_layers, dropout, **conv_kwargs):
        raise NotImplementedError

    def forward(self, data):
        """Encode a graph batch into node embeddings."""
        type_emb = self.type_embedding(data.node_type)
        x = torch.cat([data.x, type_emb], dim=-1)
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return x


class StandardGCNEncoder(_StandardGraphEncoder):
    """GCN encoder used by the GCN ablation."""

    def _build_convs(self, hidden_dim, num_layers, dropout, **conv_kwargs):
        del dropout
        del conv_kwargs
        return [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]


class StandardGATEncoder(_StandardGraphEncoder):
    """GAT encoder used by the GAT ablation."""

    def __init__(self, node_dim, hidden_dim, num_layers=2, dropout=0.1, heads=4):
        super().__init__(node_dim, hidden_dim, num_layers=num_layers, dropout=dropout, heads=heads)

    def _build_convs(self, hidden_dim, num_layers, dropout, **conv_kwargs):
        heads = conv_kwargs["heads"]
        head_out_dim = hidden_dim // heads
        return [
            GATConv(hidden_dim, head_out_dim, heads=heads, concat=True, dropout=dropout)
            for _ in range(num_layers)
        ]


class StandardSAGEEncoder(_StandardGraphEncoder):
    """GraphSAGE encoder used by the SAGE ablation."""

    def _build_convs(self, hidden_dim, num_layers, dropout, **conv_kwargs):
        del dropout
        del conv_kwargs
        return [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)]

# ==============================
# Actor Network (Policy Network)
# ==============================
class ActorNetwork(nn.Module):
    """Policy network that outputs bounded velocity commands."""
    
    def __init__(self, action_dim=3, hidden_dim=128):
        super().__init__()
        # Actor encoder
        self.actor_encoder = PhysicsInformedGNNEncoder(
            node_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # Mean head
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Log standard deviation head
        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Parameter initialization
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self._layer_init(self.mean_head[0])
        self._layer_init(self.mean_head[2], std=0.01)
        
        self._layer_init(self.log_std_head[0])
        self._layer_init(self.log_std_head[2], std=0.01)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
        
    def forward(self, state, deterministic=False):
        """Return actions and log-probabilities for active nodes."""
        node_features = self.actor_encoder(state)

        if hasattr(state, 'remain_mask'):
            remain_features = node_features[state.remain_mask]
        else:
            remain_features = node_features
        
        if len(remain_features) == 0:
            return torch.empty((0, config_dimension), device=device), torch.empty((0,), device=device)
        
        raw_mean = self.mean_head(remain_features)
        mean = torch.clamp(raw_mean, -3.0, 3.0)
        log_std = self.log_std_head(remain_features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        if deterministic:
            raw_actions = mean
            actions = torch.tanh(raw_actions) * config_max_speed
            log_probs = torch.zeros(mean.shape[0], device=device)
            log_probs = log_probs.sum(dim=-1)
        else:
            normal = torch.distributions.Normal(mean, std)
            raw_actions = normal.rsample()
            actions = torch.tanh(raw_actions) * config_max_speed
            
            log_probs = normal.log_prob(raw_actions)
            log_probs -= torch.log(1 - torch.tanh(raw_actions).pow(2) + 1e-6)
            log_probs = log_probs.sum(dim=-1)
        
        return actions, log_probs
    
    def evaluate(self, state, actions):
        """Evaluate log-probability and entropy for provided actions."""
        node_features = self.actor_encoder(state)

        if hasattr(state, 'remain_mask'):
            remain_features = node_features[state.remain_mask]
        else:
            remain_features = node_features
        
        if len(remain_features) == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        raw_mean = self.mean_head(remain_features)
        mean = torch.clamp(raw_mean, -3.0, 3.0)
        log_std = self.log_std_head(remain_features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        
        norm_actions = actions / config_max_speed
        norm_actions = torch.clamp(norm_actions, -1.0 + 1e-6, 1.0 - 1e-6)
        raw_actions = torch.atanh(norm_actions)
        
        log_prob_gauss = normal.log_prob(raw_actions).sum(dim=-1)
        correction = torch.log(1 - norm_actions.pow(2) + 1e-6).sum(dim=-1)
        
        log_probs = log_prob_gauss - correction
        
        entropy = normal.entropy().sum(dim=-1).mean()
        
        return log_probs, entropy

# ==============================
# Critic Network (Value Network)
# ==============================
# ==============================
# Ablation Baseline: Fully Decentralized Critic (IPPO Paradigm)
# ==============================
class DecentralizedCriticNetwork(nn.Module):
    """Critic ablation that estimates values from local features only."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # Use the same local encoder as the full model for a fair ablation.
        self.local_encoder = PhysicsInformedGNNEncoder(
            node_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # This critic only sees local features.
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) 
        )

    def forward(self, state):
        """Return node-wise values for active nodes."""
        h_local = self.local_encoder(state)

        if hasattr(state, 'remain_mask'):
            h_local_active = h_local[state.remain_mask]
        else:
            h_local_active = h_local
            
        if h_local_active.size(0) == 0: 
            return torch.empty(0, device=state.x.device)

        values = self.value_head(h_local_active)

        return values.squeeze(-1)

class CriticNetwork(nn.Module):
    """Centralized critic with local features plus mean/max pooled context."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        self.local_encoder = PhysicsInformedGNNEncoder(
            node_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) 
        )

    def forward(self, state):
        """Return node-wise values with pooled global context."""
        h_local = self.local_encoder(state)

        if hasattr(state, 'remain_mask'):
            h_local_active = h_local[state.remain_mask]
            batch_active = state.batch[state.remain_mask]
        else:
            h_local_active = h_local
            batch_active = state.batch
            
        if h_local_active.size(0) == 0: return torch.empty(0, device=state.x.device)

        h_global_mean = global_mean_pool(h_local_active, batch_active) 
        h_global_max = global_max_pool(h_local_active, batch_active)
        h_global = torch.cat([h_global_mean, h_global_max], dim=-1)

        h_global_broadcast = h_global[batch_active]
        h_combined = torch.cat([h_local_active, h_global_broadcast], dim=-1)

        return self.value_head(h_combined).squeeze(-1)
    
    
# ==============================
# Discriminator Network
# ==============================
class DiscriminatorNetwork(nn.Module):
    """Node-wise discriminator used to compute the GAIL reward."""

    def __init__(self, action_dim, hidden_dim=128):
        super(DiscriminatorNetwork, self).__init__()
        
        self.discriminator_encoder = PhysicsInformedGNNEncoder(
            node_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=0.0
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.discriminator_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state, action):
        """Return node-wise expert probabilities for active nodes."""
        state_encoded = self.discriminator_encoder(state)

        if hasattr(state, 'remain_mask'):
            valid_state_encoded = state_encoded[state.remain_mask]
        else:
            valid_state_encoded = state_encoded

        if action.shape[0] != valid_state_encoded.shape[0]:
            if action.dim() == 2 and action.shape[0] == 1:
                action = action.expand(valid_state_encoded.shape[0], -1)
            else:
                raise RuntimeError(
                    f"Discriminator Shape Mismatch:\n"
                    f"Active State Nodes: {valid_state_encoded.shape[0]} (from state.remain_mask)\n"
                    f"Actions Provided: {action.shape[0]}\n"
                    f"Total Graph Nodes: {state_encoded.shape[0]}\n"
                    "Fix: Ensure Discriminator receives actions only for active nodes, and aligns them correctly."
                )
             
        action_encoded = self.action_encoder(action / config_max_speed)
        combined = torch.cat([valid_state_encoded, action_encoded], dim=-1)
        node_logits = self.discriminator_head(combined)
        return torch.sigmoid(node_logits)

    
