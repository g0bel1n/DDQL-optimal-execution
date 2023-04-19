from typing import Optional, Union

import torch
import torch.nn as nn

from ddql_optimal_execution import State, StateArray

#  RMSprop optimizer


class BaseQNetLayer(nn.Module):
    def __init__(
        self,
        input_size: int = 20,
        output_size: int = 20,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
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
        super().__init__()

        self.input_head = nn.Linear(state_size, n_nodes)
        self.hidden_layers = nn.ModuleList(
            [BaseQNetLayer(n_nodes, n_nodes, nn.ReLU()) for _ in range(n_layers - 2)]
        )
        self.output_head = nn.Linear(n_nodes, action_size+1) # +1 for the "do nothing" action

    def forward(self, states: Union[State, StateArray, torch.Tensor]) -> torch.Tensor:
        x = self.input_head(states.astensor if isinstance(states, State) else states)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_head(x)
        if isinstance(states, (State, StateArray)):
            x[states["inventory"] :] = -torch.inf
        else:
            for i, state in enumerate(states):
                x[i, state[-1].long() :] = -torch.inf
        return x
