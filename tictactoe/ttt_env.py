import numpy as np


class TicTacToeEnvironment:
    """
    A class representing the Tic-Tac-Toe game environment.

    Attributes:
        board (tuple): The current state of the board as a tuple of integers.
                       0 represents an empty cell, 1 represents 'X', and -1 represents 'O'.
    """

    def __init__(self):
        """
        Initializes a new Tic-Tac-Toe game with an empty board.
        """
        self.board = tuple(np.zeros(9, dtype=int))

    def reset(self):
        """
        Resets the game to its initial state with an empty board.

        Returns:
            tuple: The reset board with all cells set to 0.
        """
        self.board = tuple(np.zeros(9, dtype=int))
        return self.board

    def step(self, action, player):
        """
        Applies an action for a given player and updates the game state.

        Args:
            action (int): The position on the board (0-8) where the player wants to place their mark.
            player (str): The player making the move, either 'X' or 'O'.

        Returns:
            tuple: The updated board state.
            int: The reward for the current move. 1 for a win, 0.1 for a draw, 0 otherwise.
            bool: A boolean indicating if the game has ended.
        """
        list_board = list(self.board)
        list_board[action] = 1 if player == 'X' else -1
        self.board = tuple(list_board)
        reward, done = self.get_reward()
        return self.board, reward, done

    def get_reward(self):
        """
        Checks the board for a winning or draw state and calculates the reward.

        Returns:
            int: The reward for the current board state. 1 for a win, 0.1 for a draw, 0 otherwise.
            bool: A boolean indicating if the game has ended.
        """
        for (i, j, k) in [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]:
            line_sum = self.board[i] + self.board[j] + self.board[k]
            if abs(line_sum) == 3:
                return 1, True
        if 0 not in self.board:
            return 0.1, True
        return 0, False
