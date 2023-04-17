from torch import Tensor

from ddql_optimal_execution import State

class Agent:

    def __init__(self, initial_budget: int, horizon:int = 100):

        self.budget = initial_budget
        self.horizon = horizon
    

    def __get_action(self, state: State) -> int:
        ...
        


    def __call__(self, state) -> int:
        return self.__get_action(state)
       
        
        