import random
import warnings

from tqdm import tqdm

from ddql_optimal_execution.agent import DDQL
from ddql_optimal_execution.environnement import MarketEnvironnement
from ddql_optimal_execution.experience_replay import ExperienceReplay

from ._warnings import MaxStepsTooLowWarning
from ddql_optimal_execution.experience_replay._warnings import (
    ExperienceReplayEmptyWarning,
)

from rich.progress import track, Progress, TaskID

from typing import Any, Optional


class ConditionalProgress(Progress):
    def __init__(self, verbose=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enabled = verbose

    def __enter__(self):
        if self._enabled:
            super().__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        if self._enabled:
            super().__exit__(*args, **kwargs)

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def add_task(
        self,
        description: str,
        start: bool = True,
        total: Optional[float] = 100,
        completed: int = 0,
        visible: bool = True,
        **fields: Any
    ) -> Optional[TaskID]:
        return (
            super().add_task(description, start, total, completed, visible, **fields)
            if self._enabled
            else None
        )

    def update(
        self,
        task_id: TaskID,
        *,
        total: Optional[float] = None,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        description: Optional[str] = None,
        visible: Optional[bool] = None,
        refresh: bool = False,
        **fields: Any
    ) -> None:
        return (
            super().update(
                task_id,
                total=total,
                completed=completed,
                advance=advance,
                description=description,
                visible=visible,
                refresh=refresh,
                **fields
            )
            if self._enabled
            else None
        )


class Trainer:
    """This class is used to train a DDQL agent in a given environment.

    Attributes
    ----------
    agent : DDQL
        The agent attribute is an instance of the DDQL class, which is a reinforcement learning algorithm
    used for decision making in an environment.
    env : MarketEnvironnement
        The `env` attribute is an instance of the `MarketEnvironnement` class, which represents the
    environment in which the agent will interact and learn. It provides the agent with information about
    the current state of the market and allows it to take actions based on that information.
    exp_replay : ExperienceReplay
        The `exp_replay` attribute is an instance of the `ExperienceReplay` class, which is a memory buffer
    that stores and retrieves experiences for reinforcement learning agents.

    Methods
    -------
    pretrain(max_steps: int = 1000, batch_size: int = 32)
        This function pretrains a DDQL agent by running random episodes, taking limit actions (sell all at the beginning or the end) and storing the experiences in an
    experience replay buffer.

    train(max_steps: int = 1000, batch_size: int = 32)
        This function trains a DDQL agent by running episodes, taking actions based on the current state of the environment, and storing the experiences in an
    experience replay buffer.

    """

    def __init__(self, agent: DDQL, env: MarketEnvironnement, **kwargs):
        """This function initializes an object with an agent, environment, and experience replay capacity.

        Parameters
        ----------
        agent : DDQL
            The agent parameter is an instance of the DDQL class, which is a reinforcement learning algorithm
        used for decision making in an environment.
        env : MarketEnvironnement
            The `env` parameter is an instance of the `MarketEnvironnement` class, which represents the
        environment in which the agent will interact and learn. It provides the agent with information about
        the current state of the market and allows it to take actions based on that information.

        """
        self.agent = agent
        self.env = env
        self.exp_replay = ExperienceReplay(capacity=kwargs.get("capacity", 10000))
        self.verbose = kwargs.get("verbose", True)

    def fill_exp_replay(self, max_steps: int = 1000, verbose: bool = True):
        """This function fills an experience replay buffer with experiences from random episodes.

        Parameters
        ----------
        max_steps : int
            The `max_steps` parameter is the maximum number of steps that the function will take before
        stopping. It is used to prevent the function from running indefinitely if the experience replay
        buffer is full.

        """

        if max_steps < self.exp_replay.capacity:
            max_steps = self.exp_replay.capacity
            warnings.warn(MaxStepsTooLowWarning(max_steps))

        p_bar = (
            tqdm(
                total=min(max_steps, self.exp_replay.capacity),
                desc="Filling experience replay buffer",
            )
            if verbose
            else None
        )

        n_steps = 0

        while (not self.exp_replay.is_full) and n_steps < max_steps:
            self.__random_border_actions(p_bar=p_bar)
            n_steps += 1

    def __random_border_actions(self, p_bar: Optional[tqdm] = None):
        """This function runs a random episode, taking limit actions (sell all at the beginning or the end) and storing the experiences in an experience replay buffer."""

        random_episode = random.randint(0, len(self.env.historical_data_series) - 1)
        sell_beginning = random.randint(0, 1)
        self.env.swap_episode(random_episode)
        distance2horizon = self.env.horizon
        while not self.env.done:
            current_state = self.env.state.copy()
            action = (
                self.env.initial_inventory
                if (
                    (distance2horizon == 1 and not sell_beginning)
                    or (distance2horizon == self.env.horizon and sell_beginning)
                )
                else 0
            )
            reward = self.env.step(action)

            distance2horizon = self.env.horizon - self.env.state["period"]
            self.exp_replay.push(
                current_state,
                action,
                reward,
                self.env.state.copy(),
                distance2horizon,
            )

            if self.verbose and p_bar is not None:
                p_bar.update(1)

    def pretrain(self, max_steps: int = 1000, batch_size: int = 32):
        """This function pretrains a DDQL agent by running random episodes, taking limit actions (sell all at the beginning or the end) and storing the experiences in an
        experience replay buffer.

        Parameters
        ----------
        max_steps : int, optional
            The maximum number of steps to pretrain the agent for.
        batch_size : int, optional
            The number of experiences to sample from the experience replay buffer at each training step.

        """
        if not isinstance(self.agent, DDQL):
            raise TypeError("The agent must be an instance of the DDQL class.")

        p_bar = (
            tqdm(total=max_steps, desc="Pretraining agent") if self.verbose else None
        )

        for _ in range(max_steps):
            self.__random_border_actions()

            self.agent.learn(self.exp_replay.get_sample(batch_size))

            if self.verbose and p_bar is not None:
                p_bar.update(1)

    def train(self, max_steps: int = 1000, batch_size: int = 32):
        """This function trains an agent using the DDQL algorithm and an experience replay buffer.

        Parameters
        ----------
        max_steps : int, optional
            max_steps is an optional integer parameter that specifies the maximum number of steps to train the
        agent for. If the number of steps taken during training exceeds this value, the training process
        will stop.

        batch_size : int, optional
            batch_size is an optional integer parameter that specifies the number of experiences to sample
        from the experience replay buffer at each training step.

        """
        if not isinstance(self.agent, DDQL):
            raise TypeError("The agent must be an instance of the DDQL class.")

        if self.exp_replay.is_empty:
            warnings.warn(ExperienceReplayEmptyWarning())
            self.fill_exp_replay(max_steps=max_steps)

        if self.verbose:
            p_bar = tqdm(
                total=min(max_steps, len(self.env.historical_data_series)),
                desc="Training agent",
            )

        n_steps = 0
        for episode in range(len(self.env.historical_data_series)):
            if n_steps > max_steps:
                break
            self.env.swap_episode(episode)
            while not self.env.done:
                current_state = self.env.state.copy()
                action = self.agent(current_state)
                reward = self.env.step(action)

                distance2horizon = self.env.horizon - self.env.state["period"]
                self.exp_replay.push(
                    current_state,
                    action,
                    reward,
                    self.env.state.copy(),
                    distance2horizon,
                )

            n_steps += 1
            if self.verbose:
                p_bar.update(1)
            self.agent.learn(self.exp_replay.get_sample(batch_size))

    def test(self, max_steps: int = 1000):
        ...
