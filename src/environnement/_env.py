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
        n_periods: int = 100,
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
                    [*self.historical_data.columns, "inventory", "reward"],
                    [*self.historical_data.iloc[0].values, initial_inventory, 0],
                )
            )
        )

        self.quadratic_penalty_coefficient = quadratic_penalty_coefficient

    def swap_episode(self, episode: int) -> None:
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
                    [*self.historical_data.columns, "inventory", "reward"],
                    [*self.historical_data.iloc[0].values, self.initial_inventory, 0],
                )
            )
        )

    def step(self, action: int) -> tuple:
        # Execute one time step within the environment

        if action > self.state["inventory"]:
            raise InvalidActionError

        self._execute_action(action)

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

        return None

    def get_trading_episodes(self) -> tuple:
        # Return the trading episodes
        return None

    def _execute_action(self, action: int) -> float:
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

        self.state["inventory"] -= action  # stays an integer
        self.state["reward"] = reward

    def reset(self) -> None:
        # Reset the state of the environment to an initial state
        self.state = State(
            dict(
                zip(
                    [*self.historical_data.columns, "inventory", "reward"],
                    [*self.historical_data.iloc[0].values, self.initial_inventory, 0],
                )
            )
        )
        self.done = False

    def __repr__(self) -> str:
        return f"MarketEnvironnement(initial_inventory={self.initial_inventory}, quadratic_penalty_coefficient={self.quadratic_penalty_coefficient}, current_episode={self.current_episode}, multi_episodes={self.multi_episodes})"
