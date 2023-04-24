import torch


class State(dict):
    """The class `State` is a subclass of the built-in `dict` class.
    It is used to store the state of the environment.

    """

    def update_state(self, **kwargs):
        """This function updates the state of an object with the key-value pairs passed as keyword arguments."""
        for k, v in kwargs.items():
            if k not in self.keys():
                raise KeyError(f"Key {k} not initialized state")
            self[k] = v

    @property
    def astensor(self):
        """This function converts a dictionary of values into a PyTorch tensor.

        Returns
        -------
            The function `astensor` is returning a PyTorch tensor that is created from the values of the
        dictionary object that the function is called on. The values are first converted to a list using the
        `values()` method, and then the list is converted to a PyTorch tensor using the `torch.Tensor()`
        function. Finally, the tensor is cast to a float data type using the `.float

        """
        return torch.Tensor(list(self.values())).float()

    def copy(self) -> "State":
        """The function returns a new State object that is a copy of the current State object.

        Returns
        -------
            The `copy` method is returning a new instance of the `State` class, which is a copy of the current
        instance.

        """
        return State(self)
