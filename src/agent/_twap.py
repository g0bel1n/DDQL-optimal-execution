from ._agent import Agent
from src import State


class TWAP(Agent):
    def _get_action(self, state: State) -> int:
        return self.budget // self.horizon
