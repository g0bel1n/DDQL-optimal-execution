from ._experience_replay import ExperienceReplay
from .agent import Agent
from ._utils import get_device
from .environnement import  MarketEnvironnement
from .state import State, StateArray

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from typing import Optional

class DDQL:

    def __init__(self, replay_size : int = 1000, quadratic_penalty_coefficient : float = 1.) -> None:
        self.device = get_device()
        print(f"Using {self.device} device")

        self.pretrain_required = True

        self.experience_replay = ExperienceReplay(replay_size)
        self.quadratic_penalty_coefficient = quadratic_penalty_coefficient





    def _execute_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass




    def _pretrain(self, n_steps: int = 1000) -> dict:
        # TODO: Implement pretraining on boundary cases 
        #  assignees: g0bel1n
        state_dict = dict()
        return state_dict

    def train(self, agent: Agent, env: MarketEnvironnement, batch_size: int = 128, lr: float = 1e-3, gamma: float = 0.99, n_epochs: int = 1000) -> None:
        if self.pretrain_required:
            self._pretrain(agent, env, batch_size, lr, gamma, n_epochs)

        trading_episodes = env.get_trading_episodes()


        agent.train()
        for trading_episod in trading_episodes:

            eps = self.initial_greediness
            for period in range(1, len(trading_episod)):
                action = self._get_action(inventory, period, eps)

                
                self.experience_replay.add(state, action, reward, next_state, done)
                state = next_state

                eps = max(eps * self.greedy_decay_rate, 0.01)

                if done:
                    break


        for _ in range(n_epochs):
            optimizer.zero_grad()
            batch = replay.sample(batch_size)
            states, actions, rewards, next_states, dones = batch[:, 0:5], batch[:, 5:6], batch[:, 6:7], batch[:, 7:12], batch[:, 12:13]
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            q_values = self.main_net(states, actions)
            next_actions = self.main_net(next_states, self.main_net(next_states, torch.ones_like(actions)))
            q_values_next = self.main_net(next_states, next_actions.detach())
            q_target = rewards + (gamma * q_values_next * (1 - dones))
            loss = nn.MSELoss()(q_values, q_target)
            loss.backward()
            optimizer.step()
        
       
        
        