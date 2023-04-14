from typing import List

import torch

from ._state import State


class StateArray:
    def __init__(self, *args) -> None:
        self.values: List[State] = list(args) if args else []
        self.n: int = len(self.values)

    def __getitem__(self, item):
        return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value

    def append(self, state):
        self.values.append(state)
        self.n += 1

    def to(self, device: str = "cpu"):
        self.values = [s.to(device) for s in self.values]

    def __len__(self):
        return self.n

    def __get_item__(self, item):
        return torch.tensor([state[item] for state in self.values]).float()

    @property
    def astensor(self):
        return torch.tensor([state.astensor for state in self.values]).float()
