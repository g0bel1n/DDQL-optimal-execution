import warnings

class ExperienceReplayEmptyWarning(UserWarning):

    def __init__(self):
        self.message = "The experience replay buffer is empty."

    def __str__(self):
        return self.message
    
