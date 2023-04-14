import torch


class State(dict):
    def update_state(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    @property
    def tensor(self):
        return torch.tensor(list(self.values()))
