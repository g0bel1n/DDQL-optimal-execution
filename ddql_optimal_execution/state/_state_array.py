from typing import List

import torch

from ._state import State


# useless
class StateArray:
    """
    A class to represent a list of states
    """

    def __init__(self, *args) -> None:
        self.values: List = list(args) if args else []
        self.n: int = len(self.values)

    def __getitem__(self, item):
        return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value

    def append(self, state):
        self.values.append(state)
        self.n += 1

    def __len__(self):
        return self.n

    def __get_item__(self, item):
        return torch.tensor([state[item] for state in self.values]).float()

    def __iter__(self):
        return iter(self.values)

    @property
    def astensor(self):
        return torch.tensor([state.astensor for state in self.values]).float()
