from ._agent import Agent
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ._neural_net import QNet
from src import State, StateArray, get_device, ExperienceDict

class DDQL(Agent):

    def __init__(self, state_dict: Optional[dict] = None,  greedy_decay_rate: float = .1, target_update_rate: int = 100, initial_greediness : float = .2, mode :str = 'train', lr:float = 1e-3, state_size : int = 5, initial_budget: int =100, horizon:int = 100, gamma :float =.99, quadratic_penalty_coefficient : float = 1.) -> None:

        super().__init__(initial_budget, horizon)

        self.device = get_device()
        print(f"Using {self.device} device")

        self.main_net = QNet(state_size=state_size, action_size=initial_budget).to(self.device)
        self.target_net = QNet(state_size=state_size, action_size=initial_budget).to(self.device)

        self.state_size = state_size

        self.gamma = gamma
    
        if state_dict is not None:
            self.main_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)

        self.greedy_decay_rate = greedy_decay_rate
        self.target_update_rate = target_update_rate
        self.greediness= initial_greediness
        self.quadratic_penalty_coefficient = quadratic_penalty_coefficient

        self.mode = mode

        self.learning_step = 0

        if self.mode == 'train':
            self.optimizer = optim.RMSprop(self.main_net.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()


    def train(self) -> None:
        self.main_net.train()
        self.mode = 'train'

    def eval(self) -> None:
        self.main_net.eval()
        self.mode = 'eval'
    

    def _get_action(self, state: State) -> torch.Tensor:

        return (
            np.random.binomial(state['inventory'], 1 / state['inventory'])
            if np.random.rand() < self.greediness and self.mode == 'train'
            else self.main_net(state).argmax().item()
        )
    
    def _update_target_net(self) -> None:
        self.target_net.load_state_dict(self.main_net.state_dict())


    def _complete_target(self, experience_batch : np.ndarray[ExperienceDict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        targets, actions , states = torch.empty(len(experience_batch)), torch.empty(len(experience_batch)), torch.empty((len(experience_batch), self.state_size))
        for i,experience in enumerate(experience_batch):#can be vectorized 

            actions[i] = experience['action']
            states[i] = experience['state'].astensor
            if experience['done'] == 1:
                
                targets[i] = experience['reward']

            elif experience['done'] == 0:
                targets[i]  = experience['reward'] + self.gamma * experience['next_state']['inventory'](experience['next_state']['Price']-experience['state']['Price']) - self.quadratic_penalty_coefficient * (experience['next_state']['inventory'])**2
            else:
                best_action =  self.main_net(experience['next_state']).argmax().item()
                targets[i]  = experience['reward'] + self.gamma * self.target_net(experience['next_state'])[int(best_action)]

        return targets, actions, states

    
    def learn(self, experience_batch : np.ndarray[ExperienceDict]) -> None:
    
        targets, actions, states  = self._complete_target(experience_batch)
        state_size = states.shape[1]
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(states, actions, targets),
            batch_size=32,
            shuffle=True,
        )

        for batch in dataloader:
            target = batch[2]
            pred = self.main_net(batch[0])[torch.arange(len(batch[0])), batch[1].long()]
            loss = self.loss_fn(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.learning_step += 1
        self.greediness = max(0.01, self.greediness * self.greedy_decay_rate)
        if self.learning_step % self.target_update_rate == 0:
            self._update_target_net()
            print(f"Target network updated at step {self.learning_step} with greediness {self.greediness:.2f}")

        

    def __call__(self, state) -> torch.Tensor:
        return self._get_action(state)
       
        
        