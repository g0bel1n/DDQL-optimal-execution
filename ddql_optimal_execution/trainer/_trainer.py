import random
import warnings

from tqdm import tqdm

from ddql_optimal_execution.agent import DDQL
from ddql_optimal_execution.environnement import MarketEnvironnement
from ddql_optimal_execution.experience_replay import ExperienceReplay

from ._warnings import MaxStepsTooLowWarning


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

        if verbose:
            pbar = tqdm(total=self.exp_replay.capacity)

        n_steps = 0
        while (not self.exp_replay.is_full) and n_steps < max_steps:
            self.__random_border_actions()
            n_steps += 1
            if verbose:
                pbar.update(1)

    def __random_border_actions(self):
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

        n_steps = 0

        while n_steps < max_steps:
            self.__random_border_actions()

            n_steps += 1
            self.agent.learn(self.exp_replay.get_sample(batch_size))

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
            self.agent.learn(self.exp_replay.get_sample(batch_size))

    def test(self, max_steps: int = 1000):
        ...
