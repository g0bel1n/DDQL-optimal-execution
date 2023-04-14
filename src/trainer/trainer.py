from src.agent._agent import Agent
from src.environnement import MarketEnvironnement

class Trainer:

    def __init__(self, agent: Agent, env: MarketEnvironnement):
        self.agent = agent
        self.env = env


    def train(self, max_steps: int = 1000):
        ...

    def test(self, max_steps: int = 1000):
        ...