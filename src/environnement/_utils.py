import torch
import numpy as np

import numpy as np
import pandas as pd
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


# TODO: implement create_fake_LOB_data
# following the same data structure outputs than in the paper
def create_fake_LOB_data(
    n_samples: int = 1000, n_features: int = 10, n_classes: int = 2
) -> tuple:
    """Creates fake LOB data for testing purposes."""
    



    pass


def fake_data(S : float = 100, r : float = 0.1, sigma : float = 0.2):
    # Creation of the dataset output
    data = pd.DataFrame(pd.date_range(start='2022-01-01 11:00:00', end='2022-01-01 13:00:00', freq = "s"), columns=['date'])
    num_points, _ = np.shape(data)

    # Simulate a Black-Scholes trajectory
    dt = 1/252/6.5/3600 # scale of a second
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * np.random.normal(size=num_points - 1)
    log_returns = np.concatenate([[0], drift + diffusion])
    log_prices = np.cumsum(log_returns)
    prices = S * np.exp(log_prices)
    
    data["price"] = prices

    return data
