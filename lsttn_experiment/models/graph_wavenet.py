import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborhoodConv(nn.Module):
    def forward(self, features, adjacency):
        return torch.einsum("bcnt,nm->bcmt", features, adjacency).contiguous()


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, support_count=3, order=2):
        super().__init__()
        self.neighborhood = NeighborhoodConv()
        self.order = order
        self.dropout = dropout
        self.projection = nn.Conv2d(
            (order * support_count + 1) * input_dim, output_dim, kernel_size=(1, 1)
        )

    def forward(self, features, supports):
        outputs = [features]
        for adjacency in supports:
            propagated = self.neighborhood(features, adjacency)
            outputs.append(propagated)
            for _ in range(2, self.order + 1):
                propagated = self.neighborhood(propagated, adjacency)
                outputs.append(propagated)
        return F.dropout(
            self.projection(torch.cat(outputs, dim=1)), self.dropout, training=self.training
        )


class GraphWaveNet(nn.Module):
    """Graph WaveNet usado como extractor corto por LSTTN."""

    def __init__(
        self,
        num_nodes,
        supports,
        dropout=0.3,
        input_dim=2,
        output_dim=64,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
    ):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.supports = supports
        self.start = nn.Conv2d(input_dim, residual_channels, kernel_size=(1, 1))

        self.node_source = nn.Parameter(torch.randn(num_nodes, 10))
        self.node_target = nn.Parameter(torch.randn(10, num_nodes))
        support_count = len(supports) + 1

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        receptive_field = 1
        for _ in range(blocks):
            dilation = 1
            for _ in range(layers):
                self.filter_convs.append(nn.Conv2d(
                    residual_channels, dilation_channels,
                    kernel_size=(1, kernel_size), dilation=(1, dilation)
                ))
                self.gate_convs.append(nn.Conv2d(
                    residual_channels, dilation_channels,
                    kernel_size=(1, kernel_size), dilation=(1, dilation)
                ))
                self.skip_convs.append(nn.Conv2d(
                    dilation_channels, skip_channels, kernel_size=(1, 1)
                ))
                self.graph_convs.append(GraphConv(
                    dilation_channels, residual_channels, dropout, support_count=support_count
                ))
                self.norms.append(nn.BatchNorm2d(residual_channels))
                receptive_field += (kernel_size - 1) * dilation
                dilation *= 2

        self.end_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_2 = nn.Conv2d(end_channels, output_dim, kernel_size=(1, 1))
        self.receptive_field = receptive_field

    def forward(self, inputs):
        # LSTTN utiliza como máximo las dos primeras características de PEMS08.
        inputs = inputs[:, :2]
        if inputs.size(3) < self.receptive_field:
            inputs = F.pad(inputs, (self.receptive_field - inputs.size(3), 0, 0, 0))
        features = self.start(inputs)
        adaptive = F.softmax(F.relu(self.node_source @ self.node_target), dim=1)
        supports = [support.to(features.device) for support in self.supports] + [adaptive]
        skip = None

        for index in range(self.blocks * self.layers):
            residual = features
            filtered = torch.tanh(self.filter_convs[index](features))
            gated = torch.sigmoid(self.gate_convs[index](features))
            features = filtered * gated

            current_skip = self.skip_convs[index](features)
            if skip is not None:
                skip = skip[..., -current_skip.size(3) :] + current_skip
            else:
                skip = current_skip

            features = self.graph_convs[index](features, supports)
            features = features + residual[..., -features.size(3) :]
            features = self.norms[index](features)

        output = F.relu(skip)
        output = F.relu(self.end_1(output))
        return self.end_2(output)
