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
    # TODO: implement inventory_action_transformer according to Appendix A.1 
    # and add tests for it in tests/test_data_utils.py (if possible) (might need to implement a function to compute the inverse of the transformation)
    batch_size, a = np.shape(inventory_action_pairs)
    #if a != 2 :
        #how do we handle errors ?
    inventory_action_pairs_transformed = np.ndarray((batch_size,2))
    for i in range(0,batch_size):
        q,x=inventory_action_pairs[i,]
        #if x > q :
            #problem
        hat_q = q/q0 - 1
        hat_x = x/q0
        r = np.sqrt(hat_q**2 + hat_x**2)
        xi = -hat_x/hat_q
        theta = np.arctan(xi)
        if theta <= np.pi/4:
            tilda_r = r *np.sqrt((xi**2+1)*(2*(np.cos(np.pi/4-theta))**2))
        else :
            tilda_r = r *np.sqrt((xi**(-2)+1)*(2*(np.cos(theta-np.pi/4))**2))
        tilda_q = -tilda_r*np.cos(theta)
        tilda_x = tilda_r*np.sin(theta)
        inventory_action_pairs_transformed[i,] = tilda_q, tilda_x
    return inventory_action_pairs_transformed
    
    pass
