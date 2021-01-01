"""Search Algos: MiniMax, AlphaBeta
"""
import time

from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
# TODO: you can import more modules, if needed


class SearchAlgos:
    def __init__(self, utility, succ, perform_move, players_score, goal=None, heuristic_f=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move
        self.players_score = players_score
        self.goal = goal
        self.h = heuristic_f

    def search(self, state, depth, maximizing_player, players_score, start_time, time_limit):
        pass


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player, players_score, start_time, time_limit):
        """Start the MiniMax algorithm.
        :param time_limit: The limit of the time allowed to use for searching
        :param start_time: The time we started the searching
        :param players_score: The score of the players
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        pos = state.get_pos() if maximizing_player else state.get_opponent_pos()
        if self.goal(state, pos, players_score):
            return self.utility(players_score, maximizing_player)
        if depth == 0:
            val = self.h(state, pos)
            return val
        if maximizing_player:
            curr_max = float('-inf')  # minus infinity
            for op in self.succ(state, pos):
                next_cell = (pos[0] + op[0], pos[1] + op[1])
                prev_val = state.board[next_cell]
                print(f'search - players_score[0] before: {players_score[0]}')
                new_state = self.perform_move(state, op, pos, players_score)
                res = self.search(new_state, depth - 1, not maximizing_player, players_score, start_time, time_limit)
                if res == -2:
                    return res  # Interrupted
                if res > curr_max:
                    curr_max = res
                self.perform_move(state, op, next_cell, players_score, prev_val)  # reversed operator
                print(f'search - players_score[0] after: {players_score[0]}')
                if time.time() - start_time > time_limit:
                    return -2  # Interrupted
            return curr_max
        else:
            curr_min = float('inf')  # infinity
            for op in self.succ(state, pos):
                next_cell = (pos[0] + op[0], pos[1] + op[1])
                prev_val = state.board[next_cell]
                new_state = self.perform_move(state, op, pos, players_score)
                res = self.search(new_state, depth - 1, not maximizing_player, players_score, start_time, time_limit)
                if res == -2:
                    return res  # Interrupted
                if res < curr_min:
                    curr_min = res
                self.perform_move(state, op, next_cell, players_score, prev_val)  # reversed operator
                if time.time() - start_time > time_limit:
                    return -2  # Interrupted
            return curr_min


class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        # TODO: erase the following line and implement this function.
        raise NotImplementedError
