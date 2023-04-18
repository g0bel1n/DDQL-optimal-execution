import os

import pandas as pd

from ddql_optimal_execution import State, Preprocessor

from ._exceptions import InvalidActionError, InvalidSwapError, EpisodeIndexError


class MarketEnvironnement:
    """
    This class represents the environment in which the agent is operating. It contains information such
    as the current time step, the agent's current position, and any other relevant information about
    the environment.

    Parameters
    ----------
    initial_inventory : int, optional
        The `initial_inventory` parameter is a float that represents the initial inventory of the agent.
        The default value is 100.0.
    data_path : str, optional
        The `data_path` parameter is a string that represents the path to the directory containing the
        data files. The default value is "../data".
    n_periods : int, optional
        The `n_periods` parameter is an integer that represents the number of periods in the data files.
        The default value is 5.
    quadratic_penalty_coefficient : float, optional
        The `quadratic_penalty_coefficient` parameter is a float that represents the coefficient of the
        quadratic penalty term in the reward function. The default value is 0.01.
    multi_episodes : bool, optional
        The `multi_episodes` parameter is a boolean that indicates whether the agent is operating in
        multi-episode mode. The default value is False.

    Attributes
    ----------
    initial_inventory : int
        The `initial_inventory` attribute is a float that represents the initial inventory of the agent.
    n_periods : int
        The `n_periods` attribute is an integer that represents the number of periods in the data files.
    current_episode : int
        The `current_episode` attribute is an integer that represents the current episode number.
    multi_episodes : bool
        The `multi_episodes` attribute is a boolean that indicates whether the agent is operating in
        multi-episode mode.
    quadratic_penalty_coefficient : float
        The `quadratic_penalty_coefficient` attribute is a float that represents the coefficient of the
        quadratic penalty term in the reward function.
    horizon : int
        The `horizon` attribute is an integer that represents the number of periods in the data files.
    preprocessor : Preprocessor
        The `preprocessor` attribute is an instance of the `Preprocessor` class, which is used to
        preprocess the data files.
    historical_data_series : List[str]
        The `historical_data_series` attribute is a list of strings that represents the paths to the
        data files.
    historical_data : pd.DataFrame
        The `historical_data` attribute is a pandas DataFrame that contains the historical data.
    state : State
        The `state` attribute is an instance of the `State` class, which represents the current state of
        the environment in which the agent is operating. It contains information such as the current
        time step, the agent's current position, and any other relevant information about the
        environment.
    done : bool
        The `done` attribute is a boolean that indicates whether the episode is over.
    """

    def __init__(
        self,
        initial_inventory: int = 100,
        data_path: str = "../data",
        n_periods: int = 5,
        quadratic_penalty_coefficient: float = 0.01,
        multi_episodes: bool = False,
    ) -> None:
        self.initial_inventory = initial_inventory
        self.n_periods = n_periods

        self.current_episode = 0
        self.multi_episodes = multi_episodes
        self.quadratic_penalty_coefficient = quadratic_penalty_coefficient
        self.horizon = n_periods

        self.preprocessor = Preprocessor(
            n_periods=n_periods, QV=True, normalize_price=True
        )

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

        self.__load_episode()

        self.state_size = len(self.state)

    def __initialize_state(self) -> None:
        """This function initializes the state of an object with historical data and an initial inventory."""
        self.state = State(
            dict(
                zip(
                    [*self.historical_data.columns, "inventory"],
                    [*self.historical_data.iloc[0].values, self.initial_inventory],
                )
            )
        )

    def __load_episode(self) -> None:
        """This function loads an episode by preprocessing historical data, initializing the state, and setting
        the "done" flag to False.

        Parameters
        ----------
        df : pd.DataFrame
            The parameter `df` is a pandas DataFrame that is being passed as an argument to the
        `__load_episode` method. However, it is not being used in the method and seems to be unnecessary.

        """
        self.historical_data = self.preprocessor(self.historical_data)
        self.__initialize_state()
        self.done = False

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

        if episode >= len(self.historical_data_series):
            raise EpisodeIndexError

        self.current_episode = episode
        self.historical_data = pd.read_csv(self.historical_data_series[episode])

        self.__load_episode()

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

    def __update_state(self, action: int) -> None:
        """This function updates the state of an inventory management environment based on a given action and
        historical data.

        Parameters
        ----------
        action : int
            The parameter `action` is an integer representing the amount of inventory to be subtracted from the
        current inventory level in the `self.state` dictionary.

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

    def get_state(self, copy=True) -> State:
        """This function returns a copy of the current state of an object if the copy parameter is True,
        otherwise it returns the current state itself.

        Parameters
        ----------
        copy, optional
            A boolean parameter that determines whether a copy of the state should be returned or the original
        state object. If copy is True, a copy of the state object is returned, otherwise the original state
        object is returned.

        Returns
        -------
            The method `get_state` is returning a copy of the current state of the object if `copy` is set to
        `True`, otherwise it returns the current state object itself. The return type is `State`.

        """

        return self.state.copy() if copy else self.state

    def __compute_reward(self, action: int) -> float:
        """This function computes the reward for a given action based on the current inventory and historical
        price data.

        Parameters
        ----------
        action : int
            The input parameter `action` is an integer representing the number of shares to sell at each
        time step.

        Returns
        -------
            The function `__compute_reward` returns a float value which represents the reward calculated based
        on the given action and the current state of the environment.

        """
        inventory = self.state["inventory"]
        intra_time_steps_prices = self.historical_data[
            self.historical_data.period == self.state["period"]
        ].Price.values
        len_ts = len(intra_time_steps_prices)
        reward = 0
        for p1, p2 in zip(intra_time_steps_prices[:-1], intra_time_steps_prices[1:]):
            inventory -= action / len_ts
            reward += (
                inventory * (p2 - p1)
                - self.quadratic_penalty_coefficient * (action / len_ts) ** 2
            )

        return reward

    def reset(self) -> None:
        """The "reset" function initializes the state and sets the "done" flag to False."""
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
