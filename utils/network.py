import numpy as np
import torch
from torch import nn
from torch import relu
from torch.optim import Adam
from typing import Tuple, List, Any
from torch.nn.functional import softmax
from torch.nn import MSELoss
from settings import settings, MODELS_DIR
import os


class A3C_Network(nn.Module):
    def __init__(self, epsilon: float, gamma: float, lr: float, model_name: str = 'global_network'):
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
        self.MSE = MSELoss()

        #Set model path
        self.MODEL_PATH = os.path.join(MODELS_DIR, f"{model_name}.pth")


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:

        x = relu(self.input_layer(x))
        x = relu(self.fc1(x))
        actor_output = self.actor_layer(x)
        value = self.critic_layer(x)

        return (actor_output, value)

    def train(self, buffor) -> List[List[Any]]:


        actor_loss = 0
        critic_loss = 0
        full_reward = 0

        #Reverse buffor
        buffor.reverse()

        for board, action, reward, value in buffor:

            #Calc full revard
            full_reward = full_reward + self.gamma * reward

            advantage = full_reward - value.item()

            logits, _ = self(board)
            log_logits = torch.log(softmax(logits, dim=-1))[action]

            actor_loss += -log_logits * advantage
            critic_loss += self.MSE(value, torch.tensor(full_reward).unsqueeze(0))

        total_loss = actor_loss + critic_loss

        #Zero gradients
        self.optimizer.zero_grad()

        #Back propagation
        total_loss.backward()

        #Transfer gradients to global model
        for global_param, local_param in zip(global_network.parameters(), self.parameters()):
            global_param._grad = local_param._grad

        self.optimizer.step()


    def save(self):
        """
        Save network
        """
        torch.save(self.state_dict(), self.MODEL_PATH)

    def load(self, traning_mode: bool = True):
        """
        Load model weights
        :return:
        """
        if os.path.exists(self.MODEL_PATH):
            self.load_state_dict(torch.load(self.MODEL_PATH))

        if not traning_mode:
            self.eval()


    def take_action(self, board: np.ndarray):
        """
        Take action. Used only to playing mode
        :return:
        """

        logits, _ = self(board)
        return softmax(board, dim=-1).item()

#Initial global network
global_network = A3C_Network(**settings)
global_network.share_memory()



