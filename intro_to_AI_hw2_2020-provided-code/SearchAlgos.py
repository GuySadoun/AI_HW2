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
        print(f'maximizing_player is {maximizing_player}')
        if self.goal():
            print(f'##search - res0 = {(self.utility(players_score), None)}')
            return self.utility(players_score), None
        if depth == 0:
            print(f'##search - res1 = {(self.h(state), None)}')
            return self.h(state), None
        direction = None
        if maximizing_player:
            max_val = float('-inf')  # infinity
            curr_idx = state.get_pos()
            print(f'##search - curr_idx = {curr_idx}')
            for op in self.succ(state, curr_idx):
                next_cell = (curr_idx[0] + op[0], curr_idx[1] + op[1])
                prev_val = state.board[next_cell]
                new_state = self.perform_move(op, True, prev_val, players_score, False)
                res = self.search(new_state, depth - 1, not maximizing_player, players_score, start_time, time_limit)
                if res[1] == 'Interrupted':
                    self.perform_move(op, True, prev_val, players_score, False)  # reversed operator
                    print(f'##search - res2 = {res}')
                    return res
                if res[0] > max_val:
                    direction = op
                    max_val = res[0]
                self.perform_move(op, False, prev_val, players_score, False)  # reversed operator
                if time.time() - start_time > time_limit:
                    return max_val, 'Interrupted'
            return max_val, direction
        else:
            min_val = float('inf')  # minus infinity
            curr_idx = state.get_opponent_pos()
            print(f'##search - curr_idx = {curr_idx}')
            for op in self.succ(state, curr_idx):
                next_cell = (curr_idx[0] + op[0], curr_idx[1] + op[1])
                prev_val = state.board[next_cell]
                new_state = self.perform_move(op, True, prev_val, players_score, True)
                res = self.search(new_state, depth - 1, not maximizing_player, players_score, start_time, time_limit)
                if res[1] == 'Interrupted':
                    self.perform_move(op, True, prev_val, players_score, True)  # reversed operator
                    print(f'##search - res3 = {res}')
                    return res
                if res[0] < min_val:
                    direction = op
                    min_val = res[0]
                self.perform_move(op, False, prev_val, players_score, True)  # reversed operator
                if time.time() - start_time > time_limit:
                    print(f'##search - res interrupted')
                    return min_val, 'Interrupted'
            return min_val, direction


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
