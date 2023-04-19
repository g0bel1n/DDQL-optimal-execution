class ReplayMemoryNotFullEnoughError(Exception):
    def __init__(self, message="Replay memory is not full enough"):
        self.message = message

    def __str__(self):
        return self.message
