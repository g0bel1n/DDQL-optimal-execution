import numpy as np
import random
import torch

from .state import State


class ExperienceReplay:
    def __init__(self, capacity: int = 10000, state_size: int = 5):
        self.capacity = capacity
        self.memory = torch.empty(
            (capacity, 2 * state_size + 3), dtype=torch.float
        )
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        # If we have reached the capacity, replace  a random element between the capacity//2 older elements
        if self.position >= self.capacity:
            deleted_row = random.randint(0, self.capacity // 2)
            self.memory[deleted_row:] = self.memory[deleted_row + 1 :]
            self.position = self.capacity - 1
        self.memory[self.position] = torch.cat(
            (
                state.astensor,
                torch.tensor([action, reward]),
                next_state.astensor,
                torch.tensor([done]),
            )
        )

        self.position += 1

    def sample(self, batch_size: int = 128):
        idxs = np.random.choice(len(self.memory), batch_size, replace=False)
        return self.memory[idxs]

    def __len__(self):
        return len(self.memory)
