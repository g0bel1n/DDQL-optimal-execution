import torch


class State(dict):
    def update_state(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    @property
    def astensor(self):
        return torch.Tensor(list(self.values())).float()

    def copy(self) -> "State":
        return State(self)
