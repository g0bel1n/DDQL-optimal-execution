class ReplayMemorySamplingError(Exception):
    def __init__(self, message="Replay memory is either  not full enough or not big enough."):
        self.message = message

    def __str__(self):
        return self.message
