from ._experience_replay import ExperienceReplay
from ._neural_net import QNet
from ._utils import get_device

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Optional

class DDQL:

    def __init__(self, state_dict: Optional[dict] = None, replay_size : int = 1000, greedy_decay_rate: float = .1, target_update_rate: int = 100, quadratic_penalty_coefficient : float = 1.) -> None:
        self.device = get_device()
        print(f"Using {self.device} device")

        self.main_net = QNet().to(self.device)
        self.target_net = QNet().to(self.device)

        self.pretrain_required = True

        if state_dict is not None:
            self.main_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            self.pretrain_required = False

        self.experience_replay = ExperienceReplay(replay_size)
        self.greedy_decay_rate = greedy_decay_rate
        self.target_update_rate = target_update_rate
        self.quadratic_penalty_coefficient = quadratic_penalty_coefficient



    def _execute_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass


    def _get_action(self, state: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        pass
            
    def _update_target_net(self) -> None:
        self.target_net.load_state_dict(self.main_net.state_dict())


    def _pretrain(self, n_steps: int = 1000) -> dict:
        #TODO: Implement pretraining on boundary cases
        # assignee: @g0bel1n
        state_dict = dict()
        return state_dict

    def train(self, replay: ExperienceReplay, batch_size: int = 128, lr: float = 1e-3, gamma: float = 0.99, n_epochs: int = 1000) -> None:
        if self.pretrain_required:
            self._pretrain()

        self.main_net.train()
        optimizer = optim.RMSprop(self.main_net.parameters(), lr=lr)

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
        
       
        
        