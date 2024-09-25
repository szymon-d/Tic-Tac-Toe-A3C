import random
from .agent import agent1, agent2  # Import the two agents.
from .board import *  # Import all board-related functions and utilities.


def play_1game():
    """
    Simulates a single game between agent1 and agent2.

    The function:
    - Initializes an empty board.
    - Randomly selects the starting player.
    - Alternates turns between the two agents.
    - Updates Q-values for the current player after each action.
    - Ends when there is a winner or no possible moves (draw).

    Returns:
    - int: 0 if it's a draw, 1 if agent1 wins, -1 if agent2 wins.
    """
    # Take an empty board at the start of the game.
    board = reset_board()

    # Randomly choose the starting player (agent1 or agent2).
    if random.uniform(0, 1) > 0.5:
        player: Agent = agent1  # Start with agent1.
    else:
        player: Agent = agent2  # Start with agent2.

    while True:
        # Check if there are no possible moves left (resulting in a draw).
        if not is_any_possible_move(board):
            # Return 0 for a draw.
            return 0

        # If it's agent1's turn, let it choose an action.
        if player is agent1:
            action = agent1.choose_action(board)
        else:
            # If it's agent2's turn, let it choose an action.
            action = agent2.choose_action(board)

        # Update the board with the chosen action.
        next_board = board.copy()  # Create a copy of the current board.
        next_board[action] = player.player  # Mark the board with the player's move.

        # Calculate the reward based on the board state and the action taken.
        reward = player.get_reward(next_board, board, action)

        # Update the player's Q-values using the Q-learning algorithm.
        loss = player.network.update_q_values(board=board,
                                              next_board=next_board,
                                              action=action,
                                              reward=reward)

        # Check if the current move resulted in a winner.
        winner = check_winner(next_board)
        if winner:
            # Return the winner (1 for agent1, -1 for agent2).
            return winner

        # Swap the player for the next turn (alternate between agent1 and agent2).
        player = agent1 if player is agent2 else agent2

        # Update the board state for the next turn.
        board = next_board


def train(epochs: int, load_pretrained: bool = False):
    """
    Trains agent1 and agent2 over a specified number of games (epochs).

    Args:
    - epochs (int): the number of training games to simulate.
    - load_pretrained (bool): if True, load pretrained models for both agents.

    This function:
    - Plays multiple games (epochs) between the two agents.
    - Tracks the number of wins for each agent and draws.
    - Optionally loads pretrained models and saves the updated models after training.

    Returns:
    - None: The function prints the win statistics and saves the agents' models.
    """
    if load_pretrained:
        # Load pretrained models for both agents if specified.
        agent1.load_model(training_mode=True)
        agent2.load_model(training_mode=True)

    # Initialize statistics for tracking wins and draws.
    winners_stats = {
        0: 0,  # Number of draws.
        1: 0,  # Number of wins for agent1.
        -1: 0  # Number of wins for agent2.
    }

    # Loop over the specified number of epochs (games).
    for epoch in range(epochs):
        print('--------')
        print(f'Epoch: {epoch}')

        # Play one game and get the result (winner or draw).
        winner = play_1game()

        # Update the statistics based on the winner.
        winners_stats[winner] += 1

    # Save the updated models for both agents after training.
    agent1.save()
    agent2.save()

    # Print the final statistics after training.
    print(winners_stats)
