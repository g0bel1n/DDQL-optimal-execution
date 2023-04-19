from ._agent import Agent
from ddql_optimal_execution import State


# The TWAP class is an agent that calculates the action based on the budget and horizon.
class TWAP(Agent):
    """
    The TWAP class inherits from the Agent class. It is an agent that calculates the action based on the budget and horizon.
    It is used to compare the performance of the DDQL agent with the performance of a simple agent that uniformly distributes
    its budget over the horizon.



    Parameters
    ----------
    initial_budget : int
        The initial budget, by default 100

    horizon : int
        The horizon, by default 100


    Attributes
    ----------
    budget : int
        The budget of the agent.

    horizon : int
        The horizon of the agent.
    """

    def get_action(self, state: State) -> int:
        """
        This function returns the action to take based on the budget and horizon values.

        Args:
          state (State): The "state" parameter is an instance of the "State" class, which represents the
        current state of the environment in which the agent is operating. It contains information such
        as the current time step, the agent's current position, and any other relevant information about
        the environment. The "__get_action"

        Returns:
          an integer value which is the result of the division of the budget attribute of the object by
        the horizon attribute of the object.
        """
        return self.budget // self.horizon
