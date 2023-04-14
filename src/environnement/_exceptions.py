class InvalidActionError(Exception):
    """Exception raised when an action is forbidden.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

