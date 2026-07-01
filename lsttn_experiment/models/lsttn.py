import torch
import torch.nn as nn

from .components import AttentionFusion, DynamicGraphConv, MLPFusion, StackedDilatedConv
from .short_term import GraphWaveNetExtractor
from .transformer import MaskedSubseriesTransformer


class LSTTNVariant(nn.Module):
    def __init__(
        self,
        transformer: MaskedSubseriesTransformer,
        num_nodes: int,
        pred_len: int,
        forward_adj: torch.Tensor,
        backward_adj: torch.Tensor,
        long_hidden: int = 16,
        period_hidden: int = 16,
        short_hidden: int = 64,
        fusion: str = "attention",
        fusion_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = transformer
        self.d_model = transformer.d_model
        self.register_buffer("forward_adj", forward_adj.float())
        self.register_buffer("backward_adj", backward_adj.float())

        self.long_extractor = StackedDilatedConv(self.d_model, long_hidden)
        self.long_projection = nn.Linear(long_hidden, self.d_model)

        self.daily_graph = DynamicGraphConv(num_nodes, self.d_model, period_hidden, dropout)
        self.weekly_graph = DynamicGraphConv(num_nodes, self.d_model, period_hidden, dropout)
        self.daily_projection = nn.Linear(period_hidden, self.d_model)
        self.weekly_projection = nn.Linear(period_hidden, self.d_model)

        supports = [forward_adj.float(), backward_adj.float()]
        self.short_extractor = GraphWaveNetExtractor(num_nodes, supports, short_hidden, dropout)
        self.short_projection = nn.Linear(short_hidden, self.d_model)

        if fusion == "attention":
            self.fusion = AttentionFusion(self.d_model, fusion_heads, dropout)
        elif fusion == "mlp":
            self.fusion = MLPFusion(self.d_model, dropout)
        else:
            raise ValueError("fusion debe ser 'attention' o 'mlp'")

        self.output = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, pred_len),
        )

    def freeze_transformer(self):
        self.transformer.eval()
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Congelado también implica desactivar su dropout durante forecasting.
        if not any(parameter.requires_grad for parameter in self.transformer.parameters()):
            self.transformer.eval()
        return self

    def forward(self, x_short, x_long, x_day, x_week):
        batch = x_long.size(0)
        representations = self.transformer(x_long, mode="inference")
        _, nodes, patches, dimension = representations.shape

        long_input = representations.reshape(batch * nodes, patches, dimension).transpose(1, 2)
        long_features = self.long_projection(self.long_extractor(long_input).reshape(batch, nodes, -1))

        # Las referencias externas se codifican en el mismo espacio del MST.
        daily_repr = self.transformer.encode_subseries(x_day, position=max(patches - 25, 0))
        weekly_repr = self.transformer.encode_subseries(x_week, position=0)
        daily = self.daily_projection(self.daily_graph(daily_repr, self.forward_adj, self.backward_adj))
        weekly = self.weekly_projection(self.weekly_graph(weekly_repr, self.forward_adj, self.backward_adj))

        long_periodic = self.fusion([long_features, daily, weekly])
        short = self.short_projection(self.short_extractor(x_short))
        return self.output(torch.cat([long_periodic, short], dim=-1))
