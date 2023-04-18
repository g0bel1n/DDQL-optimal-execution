class InvalidActionError(Exception):
    """Exception raised when an action is forbidden.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message = "Sell order is greater than inventory"):
        self.message = message

    def __str__(self):
        return self.message
    

class InvalidSwapError(Exception):
    """Exception raised when an episode is swapped while in the middle of an episode.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message = "Cannot swap episode while in the middle of an episode"):
        self.message = message

    def __str__(self):
        return self.message

class EpisodeIndexError(Exception):
    """Exception raised when an episode index is out of range.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message = "Episode index is out of range"):
        self.message = message

    def __str__(self):
        return self.message