import torch.nn as nn
import torch

from typing import Optional

#  RMSprop optimizer


class BaseQNetLayer(nn.Module):
    def __init__(
        self,
        input_size: int = 20,
        output_size: int = 20,
        activation: Optional[nn.Module] = None,
    ):
        super(BaseQNetLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x):
        return self.activation(self.fc(x))


class QNet(nn.Module):
    def __init__(self, state_size: int = 5, n_nodes: int = 20, n_layers: int = 6):
        super(QNet, self).__init__()
        self.input_head = nn.Linear(state_size + 1, n_nodes)
        self.hidden_layers = nn.ModuleList(
            [BaseQNetLayer(n_nodes, n_nodes, nn.ReLU()) for _ in range(n_layers-2)]
        )
        self.output_head = nn.Linear(n_nodes, 1)

    def forward(self, states, action):
        x = torch.cat([states, action], dim=1)
        x = self.input_head(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_head(x)
        return x
