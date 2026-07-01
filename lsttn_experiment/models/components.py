import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedDilatedConv(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        layers = []
        current = input_dim
        for dilation in (1, 2, 4, 8):
            layers.extend((
                nn.Conv1d(current, hidden_dim, 3, stride=2, dilation=dilation, padding=dilation),
                nn.GELU(),
                nn.MaxPool1d(3, stride=2, padding=1),
            ))
            current = hidden_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)[:, :, -1]


class DynamicGraphConv(nn.Module):
    def __init__(self, num_nodes: int, input_dim: int, output_dim: int, dropout: float, order: int = 2):
        super().__init__()
        self.source_embedding = nn.Parameter(torch.randn(num_nodes, 10))
        self.target_embedding = nn.Parameter(torch.randn(10, num_nodes))
        self.order = order
        self.projection = nn.Linear((1 + 3 * order) * input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def propagate(features, adjacency):
        return torch.einsum("bnh,nm->bmh", features, adjacency)

    def forward(self, features, forward_adj, backward_adj):
        adaptive = F.softmax(F.relu(self.source_embedding @ self.target_embedding), dim=1)
        outputs = [features]
        for adjacency in (forward_adj, backward_adj, adaptive):
            propagated = self.propagate(features, adjacency)
            outputs.append(propagated)
            for _ in range(2, self.order + 1):
                propagated = self.propagate(propagated, adjacency)
                outputs.append(propagated)
        return self.dropout(self.projection(torch.cat(outputs, dim=-1)))


class AttentionFusion(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, components):
        batch, nodes, dimension = components[0].shape
        tokens = torch.stack(components, dim=2).reshape(batch * nodes, len(components), dimension)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return self.norm(tokens + self.dropout(attended)).mean(dim=1).reshape(batch, nodes, dimension)


class MLPFusion(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, d_model)
        )

    def forward(self, components):
        return self.network(torch.cat(components, dim=-1))
