# Feature engineering
import torch
import numpy as np

from typing import Union


def quadratic_variations(prices: torch.Tensor) -> torch.Tensor:
    """Computes the quadratic variations of a time series.

    Args:
        prices (torch.Tensor): Time series of prices.

    Returns:
        torch.Tensor: Time series of quadratic variations.
    """
    return torch.pow(prices[1:] - prices[:-1], 2)


def inventory_action_transformer(
    inventory_action_pairs=Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    It takes in a list of inventory-action pairs, and returns a list of inventory-action pairs, where
    each inventory-action pair is transformed according to Appendix A.1


    :param inv_act_pairs: a tensor of shape (batch_size, 2) where the first column is the inventory and
    the second column is the action
    """

    # TODO: implement inventory_action_transformer according to Appendix A.1 and add tests for it in tests/test_data_utils.py (if possible) (might need to implement a function to compute the inverse of the transformation)
    pass
