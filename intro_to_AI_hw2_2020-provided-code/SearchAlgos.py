"""Search Algos: MiniMax, AlphaBeta
"""
import random
import time

from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
# TODO: you can import more modules, if needed


class SearchAlgos:
    def __init__(self, utility, succ, perform_move, players_score, goal=None, heuristic_f=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The successor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move
        self.players_score = players_score
        self.goal = goal
        self.h = heuristic_f

    def search(self, state, depth, maximizing_player):
        pass


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player, is_root=False):
        """Start the MiniMax algorithm.
        :param is_root: is it first call
        :param players_score: The score of the players
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        pos = state.get_pos() if maximizing_player else state.get_opponent_pos()
        if self.goal(state, pos):
            return self.utility(state.players_score, maximizing_player)
        if depth == 0:
            val = self.h(state, pos)
            return val
            # return val if maximizing_player else -val
        if maximizing_player:
            curr_max = float('-inf')  # minus infinity
            for op in self.succ(state, pos):
                next_cell = (pos[0] + op[0], pos[1] + op[1])
                prev_val = state.board[next_cell]
                new_state = self.perform_move(state, op, pos)
                res = self.search(new_state, depth - 1, not maximizing_player)
                if res == -2 and not is_root:
                    return res  # Interrupted
                elif is_root:
                    return curr_max
                if res > curr_max:
                    curr_max = res
                self.perform_move(state, op, next_cell, prev_val)  # reversed operator
                if state.get_time_left() < 0.7:
                    return -2  # Interrupted
            return curr_max
        else:
            curr_min = float('inf')  # infinity
            for op in self.succ(state, pos):
                next_cell = (pos[0] + op[0], pos[1] + op[1])
                prev_val = state.board[next_cell]
                new_state = self.perform_move(state, op, pos)
                res = self.search(new_state, depth - 1, not maximizing_player)
                if res == -2:
                    return res  # Interrupted
                if res < curr_min:
                    curr_min = res
                self.perform_move(state, op, next_cell, prev_val)  # reversed operator
                if state.get_time_left() < 0.7:
                    return -2  # Interrupted
            return curr_min


class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT, is_root=False):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        pos = state.get_pos() if maximizing_player else state.get_opponent_pos()
        if self.goal(state, pos):
            return self.utility(state.players_score, maximizing_player)
        if depth == 0:
            val = self.h(state, pos)
            return val
        if maximizing_player:
            curr_max = float('-inf')  # minus infinity

            # children_randomed = self.succ(state, pos).copy()
            # random.shuffle(children_randomed)
            # for op in children_randomed:
            for op in self.succ(state, pos):
                next_cell = (pos[0] + op[0], pos[1] + op[1])
                prev_val = state.board[next_cell]
                new_state = self.perform_move(state, op, pos)
                res = self.search(new_state, depth - 1, not maximizing_player, alpha, beta)
                if res == -2 and not is_root:
                    return res  # Interrupted
                elif is_root:
                    return curr_max
                if res > curr_max:
                    curr_max = res
                self.perform_move(state, op, next_cell, prev_val)  # reversed operator
                # start alpha beta adaption:
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return float('inf')
                # end alpha beta adaption
                if state.get_time_left() < 0.7:
                    return -2  # Interrupted
            return curr_max
        else:
            curr_min = float('inf')  # infinity
            for op in self.succ(state, pos):
                next_cell = (pos[0] + op[0], pos[1] + op[1])
                prev_val = state.board[next_cell]
                new_state = self.perform_move(state, op, pos)
                res = self.search(new_state, depth - 1, not maximizing_player, alpha, beta)
                if res == -2:
                    return res  # Interrupted
                if res < curr_min:
                    curr_min = res
                self.perform_move(state, op, next_cell, prev_val)  # reversed operator
                # start alpha beta adaption:
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    return float('-inf')
                # end alpha beta adaption
                if state.get_time_left() < 0.7:
                    return -2  # Interrupted
            return curr_min
