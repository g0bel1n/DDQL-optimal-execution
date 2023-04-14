from typing import List

import torch

from ._state import State


class StateArray:
    def __init__(self) -> None:
        self.values: List[State] = []
        self.n: int = 0

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

    @property
    def inventory(self):
        return torch.tensor([s["inventory"] for s in self.values])

    @property
    def price(self):
        return torch.tensor([s["price"] for s in self.values])

    @property
    def values(self):
        return torch.tensor(self.values)

    @property
    def n(self):
        return self.n
