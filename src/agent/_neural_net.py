from typing import Optional

import torch.nn as nn
import torch

from src import StateArray

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
    def __init__(
        self,
        action_size: int = 20,
        state_size: int = 5,
        n_nodes: int = 20,
        n_layers: int = 6,
    ):
        super(QNet, self).__init__()
        self.input_head = nn.Linear(state_size, n_nodes)
        self.hidden_layers = nn.ModuleList(
            [BaseQNetLayer(n_nodes, n_nodes, nn.ReLU()) for _ in range(n_layers - 2)]
        )
        self.output_head = nn.Linear(n_nodes, action_size)

    def forward(self, states: StateArray):
        x = self.input_head(states.values)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_head(x)
        for i in range(states.n):
            x[i, states[i].inventory] = -torch.inf
        return x
