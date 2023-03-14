import pandas as pd 
from ._state import State

class MarketEnvironnement:
    
    def __init__(self, initial_inventory : float = 100.) -> None:

        self.historical_data = pd.read_csv("data/historical_data.csv")
        self.historical_data = self.historical_data.set_index("Date")

        # TODO: load episodes from csv ?
        # assignees: g0bel1n



        self.horizon = self.historical_data.shape[0]

        self.current_step = 0
        self.cols = self.historical_data.columns

        initial_state = self.historical_data.iloc[self.current_step, :].values + [initial_inventory, self.current_step]
        states_elements = self.cols + ['inventory', 'step']

        self.done = False

        self.state = State(states_elements, initial_state)    

    def step(self, action: float) -> tuple:
        # Execute one time step within the environment
        self.current_step += 1

        reward = self._get_reward(action)

        self.done = self.current_step == self.horizon - 1

        self.state['inventory'] = self.state['inventory'] + action
        self.state['step'] = self.state['step'] + 1


        if not self.done:
            self.state.update_state(**self.historical_data.iloc[self.current_step, :].values)

        return None
    
    def get_trading_episodes(self) -> tuple:
        # Return the trading episodes
        return None
    
    
    def _get_reward(self, action: int) -> float:
        return self.state[-1] if action == 0 else -self.state[-1]
    
    def reset(self) -> None:
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.state = self.historical_data.iloc[self.current_step, :].values
        return self.state
    
    