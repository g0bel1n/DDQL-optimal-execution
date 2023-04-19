from torch import Tensor

from ddql_optimal_execution import State
from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, initial_budget: int, horizon:int = 100):

        self.budget = initial_budget
        self.horizon = horizon
    

    @abstractmethod
    def get_action(self, state: State) -> int:
        ...
        


    def __call__(self, state) -> int:
        return self.get_action(state)
       
        
        