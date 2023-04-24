import random

import numpy as np

from ddql_optimal_execution import State

from ._experience_dict import ExperienceDict

from ._exceptions import ReplayMemorySamplingError


# The `ExperienceReplay` class is a memory buffer that stores and retrieves experiences for
# reinforcement learning agents.
class ExperienceReplay:
    """The `ExperienceReplay` class is a memory buffer that stores and retrieves experiences for
    reinforcement learning agents.

    Attributes
    ----------
    capacity : int
        The `capacity` attribute is an integer that represents the maximum number of experiences that can be stored in the memory buffer.
    memory : np.ndarray
        The `memory` attribute is a numpy array that stores the experiences in the memory buffer.
    position : int
        The `position` attribute is an integer that represents the current position in the memory buffer.

    Methods
    -------
    __make_room()
        This function randomly deletes a row from a memory list in the first half of the list
        and shifts the remaining rows up by one
        position.
    push(state: State, action: int, reward: float, next_state: State, dist2Horizon: int)
        This is a method to add an experience tuple to a memory buffer with a fixed capacity.
    sample(batch_size: int)
        This function samples a batch of experiences from the memory buffer.
    __len__()
        This function returns the length of the memory buffer.

    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = np.empty(capacity, dtype=object)
        self.position = 0

    def __make_room(self):
        """
        This function randomly deletes a row from a memory list in the first half of the list
        and shifts the remaining rows up by one
        position.
        """
        deleted_row = random.randint(0, self.capacity // 2)
        self.memory[deleted_row:-1] = self.memory[deleted_row + 1 :]
        self.position = self.capacity - 1

    def push(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        dist2Horizon: int,
    ) -> None:
        """This is a method to add an experience tuple to a memory buffer with a fixed capacity.

        Parameters
        ----------
        state : State
          The current state of the agent, which is usually represented as a vector or an array of values that
        describe the environment.
        action : int
          The action taken by the agent in the given state.
        reward : float
          The reward parameter is a float value that represents the reward received by the agent for taking
        the action in the given state. It is used to update the Q-values of the state-action pairs in the
        reinforcement learning algorithm.
        next_state : State
          The next state is the state that the agent transitions to after taking an action in the current
        state. It is represented as an object of the State class.
        dist2Horizon : int
          dist2Horizon refers to the distance to the horizon, which is the maximum number of steps the agent
        can take before the episode ends. It is used to keep track of how many steps are left in the episode
        when storing experiences in the replay buffer.

        """

        if self.position >= self.capacity:
            self.__make_room()

        self.memory[self.position] = ExperienceDict(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "dist2Horizon": dist2Horizon,
            }
        )

        self.position += 1

    def get_sample(self, batch_size: int = 128):
        """This function returns a random sample of a specified batch size from a memory.

        Parameters
        ----------
        batch_size : int, optional
          The batch size is the number of samples that will be randomly selected from the memory buffer to be
        used for training or inference. In this case, the default batch size is 128, which means that 128
        samples will be randomly selected from the memory buffer.

        Returns
        -------
          The function `get_sample` is returning a batch of randomly selected samples from the memory. The
        size of the batch is determined by the `batch_size` parameter. The function returns an array of
        samples from the memory, where each sample is represented as a tuple of `(state, action, reward,
        next_state, done)` values.
        """

        if self.position < batch_size:
            raise ReplayMemorySamplingError

        idxs = np.random.choice(self.position, batch_size, replace=False)
        return self.memory[idxs]

    def __len__(self):
        return len(self.memory)

    @property
    def is_full(self):
        return self.position >= self.capacity

    @property
    def is_empty(self):
        return self.position == 0
