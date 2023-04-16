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



def fake_data(S : float = 100, r : float = 0.1, sigma : float = 0.2):
    """
    The function generates a fake dataset of stock prices using the Black-Scholes model.
    
    Args:
      S (float): The initial stock price. Defaults to 100
      r (float): r is the risk-free interest rate used in the Black-Scholes model to calculate the
    expected return of an asset. It represents the rate of return an investor could earn on a risk-free
    investment, such as a government bond.
      sigma (float): Sigma is the volatility of the underlying asset in the Black-Scholes model. It is a
    measure of the amount of uncertainty or risk associated with the asset's price movements over a
    given period of time. A higher value of sigma indicates a higher level of volatility and vice versa.
    
    Returns:
      The function `fake_data` returns a pandas DataFrame with two columns: "date" and "price". The
    "date" column contains a sequence of datetime values ranging from '2022-01-01 11:00:00' to
    '2022-01-01 13:00:00' with a frequency of one second. The "price" column contains simulated stock
    prices using the Black-Scholes model.
    """
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
