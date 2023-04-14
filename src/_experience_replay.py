import numpy as np
import random

from .environnement._state import State


class ExperienceReplay:
    def __init__(
        self, capacity: int = 10000, transition_size: int = 5
    ):
        self.capacity = capacity
        self.memory = np.zeros(capacity, transition_size, dtype=State)  # check if this is correct
        self.position = 0

    def __getitem__(self, index) -> State:
        return self.memory[index]

    def __setitem__(self, index, value: State):
        self.memory[index] = value

    def __iter__(self):
        return iter(self.memory)

    def __next__(self):
        return next(self.memory)

    def push(self, transition: State):
        """Saves a transition."""
        # If we have reached the capacity, replace  a random element between the capacity//2 older elements
        if self.position >= self.capacity:
            deleted_row = random.randint(0, self.capacity // 2)
            self.memory[deleted_row:] = self.memory[deleted_row + 1 :]
            self.position = self.capacity - 1
        self.memory[self.position] = transition
        self.position += 1

    def sample(self, batch_size: int = 128):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
