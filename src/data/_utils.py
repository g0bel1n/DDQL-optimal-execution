import torch
import numpy as np

import numpy as np
def create_fake_prices(n_samples:int =1000, mean : float = 10., std : float = 1., return_type: str = 'numpy') -> torch.Tensor:
    increments = np.random.normal(0, 1, n_samples)
    prices = np.exp(np.cumsum(increments))
    prices = (prices - prices.mean()) / prices.std()
    prices = prices * std + mean
    if return_type == 'numpy':
        return prices
    elif return_type == 'torch':
        return torch.tensor(prices, dtype=torch.float32)
    else:
        raise ValueError("return_type must be either 'numpy' or 'torch'")



def create_fake_LOB_data(
    n_samples: int = 1000, n_features: int = 10, n_classes: int = 2
) -> tuple:
    """Creates fake LOB data for testing purposes."""
    pass