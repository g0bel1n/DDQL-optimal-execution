import pandas as pd 

class MarketEnvironnement:
    
    def __init__(self) -> None:

        self.historical_data = pd.read_csv("data/historical_data.csv")
        self.historical_data = self.historical_data.set_index("Date")

        self.horizon = self.historical_data.shape[0]

        self.current_step = 0
        self.state = self.historical_data.iloc[self.current_step, :].values 

    def step(self, action: int) -> tuple:
        # Execute one time step within the environment
        self.current_step += 1

        reward = self._get_reward(action)

        done = self.current_step == self.horizon - 1

        self.state = self.historical_data.iloc[self.current_step, :].values

        return self.state, reward, done
    
    def _get_reward(self, action: int) -> float:
        return self.state[-1] if action == 0 else -self.state[-1]
    
    def reset(self) -> None:
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.state = self.historical_data.iloc[self.current_step, :].values
        return self.state
    
    