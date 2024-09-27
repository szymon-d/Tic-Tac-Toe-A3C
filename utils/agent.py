import torch

from network import A3C_Network
from settings import *  # Import settings for neural network configuration.
from environment import *  # Import necessary board functions.
import numpy as np
import random
from torch import multiprocessing as mp
import os
from torch.nn import functional as F

class Agent(mp.Process):
    """
    This class represents a reinforcement learning agent that interacts with the environment (the game board),
    decides its moves based on a neural network, and learns through reward-based feedback.
    """

    def __init__(self,
                 global_network: A3C_Network,
                 games_played: mp.Value,
                 games2play: int,
                 *args, **kwargs):
        """
        Initialize the Agent with a player number and create its QNetwork.

        Args:
        - player (int): Player number, either 1 (for player 1) or -1 (for player 2).
        """
        super().__init__(*args, **kwargs)
        self.global_network = global_network

        self.games2play = games2play
        self.games_played = games_played

    def __str__(self):
        """
        String representation of the agent.

        Returns:
        - str: "Agent X" or "Agent O", depending on the player number.
        """
        return f"Agent {mapper[self.player]}"

    def get_reward(self, board: np.ndarray, board_before_action: np.ndarray, action: int, player: int) -> float:
        """
        Compute the reward for the agent based on the current state of the board and the action taken.

        Args:
        - board (np.ndarray): The game board after the agent's action.
        - board_before_action (np.ndarray): The game board before the agent's action.
        - action (int): The action (position) taken by the agent on the board.

        Returns:
        - float: The reward value based on the game state and action.
        """
        # Check if there is a winner after the agent's action.
        winner = check_winner(board=board)

        if winner:
            return winner * player  # Reward if the agent won (or punishment if the opponent won).

        # Reward for starting from the middle (favorable strategy).
        if np.sum(board == 0) == 8 and board.flatten().tolist()[4] == player:
            return 0.9

        # Punishment for not starting from the middle.
        if np.sum(board == 0) == 8 and board.flatten().tolist()[4] != player and np.sum(board == player * -1) == 0:
            return -0.9

        # Punishment if the agent missed a chance to win.
        if if_player_win_in_next_turn(board=board_before_action, player=player, amount_of_winnig_configs=1):
            return -1

        # Punishment if the agent failed to block an opponent's winning move.
        if if_player_lost_in_next_turn(board=board, player=player):
            return -1

        # Reward if the agent can win in the next turn (two possible configurations).
        if if_player_win_in_next_turn(board=board, player=player, amount_of_winnig_configs=2):
            return 0.6

        # Smaller reward if the agent can win in the next turn (one possible configuration).
        if if_player_win_in_next_turn(board=board, player=player, amount_of_winnig_configs=1):
            return 0.3

        # If the board is full, it’s a draw.
        if np.sum(board == 0) == 0:
            return 0

        # Small reward for making a move in a corner position (strategically important).
        if action in (0, 2, 6, 8):
            return 0.1

        # Default no reward for a neutral move.
        return 0


    def run(self):
        while self.games_played.value < self.games2play:
            self.play1game()

    def play1game(self):

        #Initial buffor
        buffor = []

        #Initial local network as copy of global network
        local_network = A3C_Network(**settings)
        local_network.load_state_dict(state_dict=self.global_network.state_dict())

        # Take empty board
        board = reset_board()

        player = 1 #1 or -1

        # Check if more game is required
        while any([not check_winner(board=board), is_any_possible_move(board=board)]):

            #Get available actions
            available_action = get_available_action(board)


            if player == 1:
                #Convert np.ndarray to tensor
                tensor_board = torch.FloatTensor(board)

                #Take logits (action output) and value (critic output)
                logits, value = local_network(tensor_board)

                probs = F.softmax(logits, dim=-1)
                #Mask not available action (set in 0). That will make it will be never draw
                f = lambda x: 1 if x in available_action else 0
                mask = torch.tensor([f(i) for i in range(9)], dtype=torch.bool)

                probs = torch.where(mask, probs, torch.tensor(0.0))

                #Draw choice but do this based on probability.
                #In most cases the best propability will be choices but in some cases it will not
                #That is exploration machine
                action = torch.multinomial(probs, num_samples=1)

                next_board = board.copy()
                next_board[action] = player

                #Take reward for this action
                reward = self.get_reward(board=next_board,
                                         board_before_action=board,
                                         action=action,
                                         player=player)
                #Update buffor
                buffor.append([board, action, reward, value])

                print(buffor)
                if len(buffor) == 5:
                    local_network.train(buffor)
                    #Clear buffor
                    buffor.clear()

                board = next_board
            else:
                #Random choice
                random_choice = random.choices(available_action)
                board[random_choice] = player

            # Swap players
            player *= -1

        #Increase number of done games
        with self.games_played.get_lock():
            self.games_played.value += 1





    # def save(self):
    #     """
    #     Save the agent's Q-network model to the specified file path.
    #     """
    #     torch.save(self.network.state_dict(), self.model_path)
    #
    # def load_model(self, training_mode: bool = False):
    #     """
    #     Load the agent's Q-network model from the specified file path.
    #
    #     Args:
    #     - training_mode (bool): If True, the network is set to training mode; otherwise, it's set to evaluation mode.
    #     """
    #     # Reinitialize the network and load the saved model.
    #     self.network = QNetwork(**settings[self.player])
    #     self.network.load_state_dict(torch.load(self.model_path))
    #
    #     # Set the network to evaluation mode unless training mode is explicitly requested.
    #     if not training_mode:
    #         self.network.eval()
