import torch.nn as nn

from .graph_wavenet import GraphWaveNet


class GraphWaveNetExtractor(nn.Module):
    """Rama corta usada por la implementación concreta del paper."""

    def __init__(self, num_nodes: int, supports, output_dim: int, dropout: float):
        super().__init__()
        self.register_buffer("support_forward", supports[0].float())
        self.register_buffer("support_backward", supports[1].float())
        self.network = GraphWaveNet(
            num_nodes=num_nodes,
            supports=[self.support_forward, self.support_backward],
            dropout=dropout,
            input_dim=2,
            output_dim=output_dim,
            residual_channels=32,
            dilation_channels=32,
            skip_channels=256,
            end_channels=512,
            kernel_size=2,
            blocks=4,
            layers=2,
        )

    def forward(self, inputs):
        # [B,T,N,C] -> [B,C,N,T]
        self.network.supports = [self.support_forward, self.support_backward]
        features = inputs.permute(0, 3, 2, 1)
        output = self.network(features)
        return output.squeeze(-1).transpose(1, 2)
