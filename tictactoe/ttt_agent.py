import numpy as np
import random


class TicTacToeAgent:
    """
    A Q-learning agent for playing Tic-Tac-Toe.

    Attributes:
        alpha (float): The learning rate.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The exploration rate for choosing random actions.
        q_table (dict): A dictionary to store state-action values (Q-values).
    """

    def __init__(self, alpha=0.01, gamma=0.9, epsilon=0.5):
        """
        Initializes the agent with the given parameters.

        Args:
            alpha (float): The learning rate. Default is 0.01.
            gamma (float): The discount factor for future rewards. Default is 0.9.
            epsilon (float): The exploration rate for choosing random actions. Default is 0.5.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def decrease_exploration(self):
        """
        Decreases the exploration rate slightly.

        This method is typically called after each episode to gradually reduce
        the exploration rate, encouraging the agent to exploit learned policies
        rather than exploring new actions.
        """
        self.epsilon *= 0.9999

    @staticmethod
    def get_possible_actions(board):
        """
        Returns a list of possible actions on the current board.

        Args:
            board (list or tuple): The current state of the board as a list or tuple of integers.
                                   0 represents an empty cell, 1 represents 'X', and -1 represents 'O'.

        Returns:
            list: A list of indices representing empty cells on the board.
        """
        return [i for i, v in enumerate(board) if v == 0]

    def choose_action(self, state):
        """
        Chooses an action based on the current state using an epsilon-greedy policy.

        Args:
            state (tuple): The current state of the board as a tuple of integers.

        Returns:
            int: The index of the chosen action (0-8).
        """
        possible_actions = self.get_possible_actions(state)
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_actions)

        q_values = [self.q_table.get((state, action), 0) for action in possible_actions]
        max_q = max(q_values)
        return random.choice([action for action, q_value in zip(possible_actions, q_values) if q_value == max_q])

    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-values based on the current action and the received reward.

        Args:
            state (tuple): The state of the board before the action.
            action (int): The index of the action taken (0-8).
            reward (float): The reward received after taking the action.
            next_state (tuple): The state of the board after taking the action.
        """
        next_possible_actions = self.get_possible_actions(next_state)
        next_q_values = [self.q_table.get((next_state, a), 0) for a in next_possible_actions]
        best_next_q = max(next_q_values) if next_q_values else 0
        current_q = self.q_table.get((state, action), 0)
        self.q_table[(state, action)] = current_q + self.alpha * (reward - self.gamma * best_next_q - current_q)
