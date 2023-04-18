class MaxStepsTooLowWarning(UserWarning):
    """Warning raised when the max steps is too low."""

    def __init__(self, value):
        self.message = f"Max steps is too low. It is set to {value}."

    def __str__(self):
        return self.message
