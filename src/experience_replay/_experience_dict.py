# The class `ExperienceDict` is a subclass of the built-in `dict` class that overrides the `__repr__`
# method to return a string representation of the dictionary with the class name included.
class ExperienceDict(dict):
    def __repr__(self) -> str:
        return f"ExperienceDict({super().__repr__()})"