from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ._neural_net import QNet
from src import State, StateArray, get_device

class Agent:

    def __init__(self, initial_budget: int, horizon:int = 100):

        self.budget = initial_budget
        self.horizon = horizon
    

    def _get_action(self, state: State) -> torch.Tensor | int:
        ...
        


    def __call__(self, state) -> torch.Tensor | int:
        return self._get_action(state)
       
        
        