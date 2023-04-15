import random

import numpy as np

from src import State

from ._experience_dict import ExperienceDict


# The `ExperienceReplay` class is a memory buffer that stores and retrieves experiences for
# reinforcement learning agents.
class ExperienceReplay:
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

    def push(self, state : State, action : int, reward : float, next_state :State, dist2Horizon: int) -> None:
        """
        This is a method to add an experience tuple to a memory buffer.
        
        Args:
          state (State): The current state of the environment at the time the experience was observed.
          action (int): The action taken by the agent in the given state.
          reward (float): The reward parameter is a float value that represents the reward received by the
        agent for taking the action in the given state.
          next_state (State): "next_state" is a variable that represents the state that the agent
        transitions to after taking an action in the current state. In other words, it is the state that the
        agent observes after performing an action in the current state.
          dist2Horizon (int): The "dist2Horizon" parameter is an int that is 0 if the state is terminal, 1 if it is pre-terminal and 2 otherwise
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
        """
        This function returns a random sample of a specified batch size from a memory array.
        
        Args:
          batch_size (int): The batch size is an integer parameter that determines the number of samples to
        be randomly selected from the memory buffer. Defaults to 128
        
        Returns:
          The function `get_sample` is returning a batch of randomly selected samples from the memory buffer
        of the agent. The size of the batch is specified by the `batch_size` parameter. The function returns
        an array of shape `(batch_size, 5)` where each row represents a sample consisting of 5 elements:
        state, action, reward, next_state, and done flag.
        """
        idxs = np.random.choice(len(self.memory), batch_size, replace=False)
        return self.memory[idxs]

    def __len__(self):
        return len(self.memory)
