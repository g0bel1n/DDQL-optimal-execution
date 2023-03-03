import torch
import random


class ExperienceReplay:
    def __init__(
        self, capacity: int = 10000, device: str = "cpu", transition_size: int = 5
    ):
        self.capacity = capacity
        self.memory = torch.zeros(capacity, transition_size)
        self.position = 0
        self.to(device)

    def __getitem__(self, index):
        return self.memory[index]

    def __setitem__(self, index, value):
        self.memory[index] = value

    def __iter__(self):
        return iter(self.memory)

    def __next__(self):
        return next(self.memory)

    def to(self, device: str = "cpu"):
        self.memory = self.memory.to(device)

    def push(self, transition):
        """Saves a transition."""
        # If we have reached the capacity, replace  a random element between the capacity//2 older elements
        if self.position >= self.capacity:
            deleted_row = random.randint(self.capacity // 2)
            self.memory[deleted_row:] = self.memory[deleted_row + 1 :]
            self.position = self.capacity - 1
        self.memory[self.position] = transition
        self.position += 1

    def sample(self, batch_size: int = 128):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
