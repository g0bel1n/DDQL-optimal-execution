import os
from typing import List

import numpy as np
import pandas as pd

from src import State

from ._exceptions import InvalidActionError, InvalidSwapError


class MarketEnvironnement:
    def __init__(
        self,
        initial_inventory: float = 100.0,
        data_path: str = "../data",
        n_periods: int = 5,
        quadratic_penalty_coefficient: float = 0.01,
        multi_episodes: bool = False,
    ) -> None:
        # multi episodes
        self.initial_inventory = initial_inventory
        self.n_periods = n_periods

        self.current_episode = 0
        self.multi_episodes = multi_episodes

        if multi_episodes:
            self.historical_data_series = []
            for file in os.listdir(data_path):
                if file.endswith(".csv"):
                    self.historical_data_series.append(f"{data_path}/{file}")

            self.historical_data = pd.read_csv(
                self.historical_data_series[self.current_episode]
            )

        else:
            self.historical_data = pd.read_csv(f"{data_path}/historical_data.csv")
        self.historical_data = self.historical_data.set_index("Date")
        _date_splits = np.split(self.historical_data.index, n_periods)
        self.historical_data["period"] = 0
        self.historical_data["period"] = self.historical_data["period"].astype(int)
        for i, split in enumerate(_date_splits):
            self.historical_data.loc[split, "period"] = i

        # agent move along Ts while reward along ts
        self.horizon = len(_date_splits)

        self.done = False

        self.state = State(
            dict(
                zip(
                    [*self.historical_data.columns, "inventory"],
                    [*self.historical_data.iloc[0].values, initial_inventory],
                )
            )
        )

        self.quadratic_penalty_coefficient = quadratic_penalty_coefficient

        self.state_size = len(self.state)

    def swap_episode(self, episode: int) -> None:
        """
        This function swaps the current episode in a time series environment and updates the historical
        data, periods, and state accordingly.
        
        Args:
          episode (int): The episode parameter is an integer that represents the episode number to be
        swapped to.
        """
        if self.state["period"] >= 1 and not self.done:
            raise InvalidSwapError

        self.current_episode = episode
        self.historical_data = pd.read_csv(self.historical_data_series[episode])
        self.historical_data = self.historical_data.set_index("Date")
        _date_splits = np.split(self.historical_data.index, self.n_periods)
        self.historical_data["period"] = 0

        for i, split in enumerate(_date_splits):
            self.historical_data.loc[split, "period"] = i

        # agent move along Ts while reward along ts
        self.horizon = len(_date_splits)

        self.done = False

        self.state = State(
            dict(
                zip(
                    [*self.historical_data.columns, "inventory"],
                    [*self.historical_data.iloc[0].values, self.initial_inventory],
                )
            )
        )

    def step(self, action: int) -> float:
        """
        This function executes one time step within the environment, raises an error if the action is
        invalid, executes the action, updates the state, and returns the reward.
        
        Args:
          action (int): an integer representing the action taken by the agent in the environment. In this
        case, it is assumed that the agent is making a decision about how much of a certain item to
        sell, and the action parameter represents the quantity of that item to sell.
        
        Returns:
          a float value, which is the reward obtained after executing the action in the environment.
        """
        # Execute one time step within the environment

        if action > self.state["inventory"]:
            raise InvalidActionError

        reward = self.__compute_reward(action)

        self.__update_state(action)

        return reward
    
    def __update_state(self, action:int) -> None:
        """
        This function updates the state of an inventory management environment based on a given action and
        historical data.
        
        Args:
          action (int): The parameter `action` is an integer representing the amount of inventory to be
        subtracted from the current inventory level in the `self.state` State object.
        """
  
        self.state["inventory"] -= action  

        self.state["period"] = self.state["period"] + 1

        self.done = (self.state["period"] == self.horizon - 1) or (
            self.state["inventory"] == 0
        )

        if not self.done:
            self.state.update_state(
                **self.historical_data[
                    self.historical_data.period == self.state["period"]
                ]
                .iloc[0]
                .to_dict()
            )

    def __compute_reward(self, action: int) -> float:
        """
        This function computes the reward for a given action based on the current inventory and historical
        price data.
        
        Args:
          action (int): The input parameter `action` is an integer representing the number of shares to sell at each time step.
        
        Returns:
          The function `__compute_reward` returns a float value which represents the reward calculated based
        on the given action and the current state of the environment.
        """
        inventory = self.state["inventory"]
        intra_time_steps = self.historical_data[
            self.historical_data.period == self.state["period"]
        ].Price.values
        len_ts = len(intra_time_steps)
        reward = 0
        for p1, p2 in zip(intra_time_steps[:-1], intra_time_steps[1:]):
            inventory -= action / len_ts
            reward += (
                inventory * (p2 - p1)
                - self.quadratic_penalty_coefficient * (action / len_ts) ** 2
            )

        return reward

    def reset(self) -> None:
        """
        The "reset" function reinitializes the environment to its initial state.
        """
        self.state = State(
            dict(
                zip(
                    [*self.historical_data.columns, "inventory"],
                    [*self.historical_data.iloc[0].values, self.initial_inventory],
                )
            )
        )
        self.done = False

    def __repr__(self) -> str:
        return f"MarketEnvironnement(initial_inventory={self.initial_inventory}, quadratic_penalty_coefficient={self.quadratic_penalty_coefficient}, current_episode={self.current_episode}, multi_episodes={self.multi_episodes})"
