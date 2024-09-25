import torch
import numpy as np
from settings import mapper  # Import the mapper dictionary to display board symbols (e.g., X, O).


def reset_board() -> np.ndarray:
    """
    Reset the game board to an empty state.

    Returns:
    - np.ndarray: A numpy array of zeros representing an empty board (1x9).
    """
    return np.zeros(9)


def is_any_possible_move(board: torch.Tensor) -> bool:
    """
    Check if there are any available moves left on the board.

    Args:
    - board (torch.Tensor): The current game board as a tensor.

    Returns:
    - bool: True if there are empty spots (zeros) on the board, False otherwise.
    """
    # Check if there are any zeros (empty spots) on the board.
    return True if np.where(board.flatten() == 0)[0].shape[0] > 0 else False


def check_winner(board: torch.Tensor) -> int:
    """
    Check if there is a winner on the current board.

    Args:
    - board (torch.Tensor): The current game board as a tensor.

    Returns:
    - int: 1 if player 1 (represented by 1) wins,
           -1 if player -1 wins,
           0 if no winner is found (game is ongoing or a draw).
    """
    # Convert the board to a flattened list (1D).
    board = board.flatten().tolist()

    # Define all possible winning configurations (rows, columns, diagonals).
    winning_configs = ((0, 1, 2),  # Row 1
                       (3, 4, 5),  # Row 2
                       (6, 7, 8),  # Row 3
                       (0, 3, 6),  # Column 1
                       (1, 4, 7),  # Column 2
                       (2, 5, 8),  # Column 3
                       (0, 4, 8),  # Diagonal 1
                       (2, 4, 6))  # Diagonal 2

    # Check if any of the winning configurations has been achieved.
    for config in winning_configs:
        if board[config[0]] == board[config[1]] == board[config[2]] != 0:
            return board[config[0]]  # Return the player number (1 or -1).

    # Return 0 if no winner is found.
    return 0


def if_player_lost_in_next_turn(board: torch.Tensor, player: int) -> bool:
    """
    Check if the given player will lose in the next turn.

    Args:
    - board (torch.Tensor): The current game board as a tensor.
    - player (int): The current player's number (1 or -1).

    Returns:
    - bool: True if the player is at risk of losing in the next move, False otherwise.
    """
    # Change the player to the opponent (multiply by -1).
    player *= -1

    # Flatten the board for easier processing.
    board = board.flatten().tolist()

    # Define winning configurations (rows, columns, diagonals).
    winning_configs = ((0, 1, 2),
                       (3, 4, 5),
                       (6, 7, 8),
                       (0, 3, 6),
                       (1, 4, 7),
                       (2, 5, 8),
                       (0, 4, 8),
                       (2, 4, 6))

    # Check if the opponent has two in a row with one empty spot in any configuration.
    for config in winning_configs:
        # Extract the board values for the current configuration.
        choices = [i for idx, i in enumerate(board) if idx in config]

        # If the opponent has two spots and one empty, the player might lose in the next turn.
        if choices.count(player) == 2 and choices.count(0) == 1:
            return True  # Player is in danger of losing.

    return False  # No immediate threat of losing.


def if_player_win_in_next_turn(board: torch.Tensor, player: int, amount_of_winnig_configs: int) -> bool:
    """
    Check if the player can win in the next move by completing a winning configuration.

    Args:
    - board (torch.Tensor): The current game board as a tensor.
    - player (int): The current player's number (1 or -1).
    - amount_of_winnig_configs (int): The number of configurations that would allow the player to win.

    Returns:
    - bool: True if the player can win in the next turn, False otherwise.
    """
    # Flatten the board for easier processing.
    board = board.flatten().tolist()

    # Define winning configurations (rows, columns, diagonals).
    winning_configs = ((0, 1, 2),
                       (3, 4, 5),
                       (6, 7, 8),
                       (0, 3, 6),
                       (1, 4, 7),
                       (2, 5, 8),
                       (0, 4, 8),
                       (2, 4, 6))

    winning_config = 0  # Counter for the number of possible winning configurations.

    # Check each configuration to see if the player is one move away from winning.
    for config in winning_configs:
        choices = [i for idx, i in enumerate(board) if idx in config]
        if choices.count(player) == 2 and choices.count(0) == 1:
            winning_config += 1  # Increment if player has two in a row with one empty spot.

    # Return True if the number of potential winning configurations is greater than or equal to the target.
    return winning_config >= amount_of_winnig_configs


def display_board(board: torch.Tensor) -> np.ndarray:
    """
    Display the board with human-readable symbols.

    Args:
    - board (torch.Tensor): The current game board as a tensor (with 1, -1, 0 values).

    Returns:
    - np.ndarray: The board as a 3x3 numpy array with 'X' for player 1, 'O' for player -1, and '-' for empty spots.
    """
    # Flatten the board and map the integer values to symbols using the 'mapper' dictionary.
    board = board.flatten().tolist()
    board = [mapper.get(int(i)) for i in board]

    # Reshape the board to a 3x3 numpy array for display purposes.
    return np.array(board, dtype=str).reshape(3, 3)
