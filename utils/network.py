import numpy as np
import torch
from torch import nn
from torch import relu
from torch.optim import Adam
from typing import Tuple, List, Any


class A3C_Network(nn.Module):
    def __init__(self, epsilon: float, gamma: float, lr: float):
        super().__init__()
        self.input_layer = nn.Linear(9, 128)
        self.fc1 = nn.Linear(128, 128)
        self.actor_layer = nn.Linear(128, 9)
        self.critic_layer = nn.Linear(128, 1)

        # Store epsilon and gamma parameters.
        self.epsilon = epsilon
        self.gamma = gamma

        # Adam optimizer for updating the network's weights with the specified learning rate.
        self.optimizer = Adam(params=self.parameters(), lr=lr)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:

        x = relu(self.input_layer(x))
        x = relu(self.fc1(x))
        actor_output = self.actor_layer(x)
        value = self.critic_layer(x)

        return (actor_output, value)

    def train(self, buffor) -> List[List[Any]]:

        for next_board, action, reward, value in buffor:
            print(reward)



