from typing import Optional

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim

from ._neural_net import QNet
from ._state import State
from ._utils import get_device


class Agent:

    def __init__(self, state_dict: Optional[dict] = None,  greedy_decay_rate: float = .1, target_update_rate: int = 100, initial_greediness : float = .2, mode :str = 'train', lr:float = 1e-3) -> None:

        self.device = get_device()
        print(f"Using {self.device} device")

        self.main_net = QNet().to(self.device)
        self.target_net = QNet().to(self.device)

    
        if state_dict is not None:
            self.main_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)

        self.greedy_decay_rate = greedy_decay_rate
        self.target_update_rate = target_update_rate
        self.greediness= initial_greediness

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
            else self.main_net(state).max().item()
        )
    
    def _update_target_net(self) -> None:
        self.target_net.load_state_dict(self.main_net.state_dict())


    def _complete_target(self, experience_batch : torch.Tensor) -> torch.Tensor:
        ids = torch.cat(torch.where(experience_batch['done'] == 0)[0], torch.where(experience_batch['predone'] == 0)[0])
        for experience in experience_batch[ids]:#can be vectorized 
            best_action =  self.main_net(experience['next_state']).argmax().item()
            target_complement = experience['gamma'] * self.target_net(experience['next_state'])[best_action]
            experience['target'] += target_complement

        return experience_batch

    
    def learn(self, experience_batch : torch.Tensor) -> None:
        experience_batch = self._complete_target(experience_batch)
        dataloader = torch.utils.data.DataLoader(experience_batch, batch_size=32, shuffle=True)
        for batch in dataloader:
            target = batch['target'].unsqueeze(1)
            pred = self.main_net(batch['state'], batch['action'])
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
       
        
        