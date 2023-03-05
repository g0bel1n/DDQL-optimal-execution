import torch

class State:

    def __init__(self, cols, values):
        
        self.idxs = {k: v for v, k in enumerate(cols)}
        self.state = torch.tensor(values)

    def _get_slice(self, key):
        return slice(self.idxs[key], self.idxs[key] + 1)

    def update_state(self, **kwargs):
        for k, v in kwargs.items():
            self.state[self._get_slice(k)] = v

    def __getitem__(self, item):
        return self.state[self._get_slice(item)]
    
    def __setitem__(self, key, value):
        self.state[self._get_slice(key)] = value
    
    def to(self, device: str = "cpu"):
        self.state = self.state.to(device)
