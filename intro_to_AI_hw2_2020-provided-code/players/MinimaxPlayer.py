"""
MiniMax Player
"""
import time

import numpy as np

import utils
from Game import Game
from SearchAlgos import MiniMax
from players.AbstractPlayer import AbstractPlayer


# TODO: you can import more modules, if needed


class State:
    def __init__(self, board, players_score, player_number):
        self.board = board
        self.players_score = players_score
        self.player_number = player_number
        # self.penalty_score = penalty_score
        # self.maximizing_player = maximizing_player
        # self.time = time
        # self.time_limit = time_limit

    def state_options(self, pos):
        num_ops_available = 0
        for d in utils.get_directions():
            i = pos[0] + d[0]
            j = pos[1] + d[1]
            if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and (self.board[i][j] not in [-1, 1, 2]):
                num_ops_available += 1
        return num_ops_available

    def reachable_white_cells(self, player_number):
        if player_number == 1:
            pos = self.get_pos()
        else:
            assert player_number == 2
            pos = self.get_opponent_pos()

        board_copy = np.copy.deepcopy(self.board)
        reachable = [pos]
        count = 0
        while len(reachable) > 0:
            curr_pos = reachable.pop()
            for d in utils.get_directions():
                i = curr_pos[0] + d[0]
                j = curr_pos[1] + d[1]

                if 0 <= i < len(board_copy) and 0 <= j < len(board_copy[0]) and (board_copy[i][j] not in [-1, 1, 2]):
                    board_copy[i, j] = -1
                    reachable.append((i, j))
                    count += 1
                    # limit the amount of white cells we want to check because far white cells are less relevant for us,
                    # since there is a high probability that they will be grey until we reach them
                    if count > board_copy.size / 2:
                        break
        return count

    def get_board(self):
        return self.board

    def get_players_score(self):
        return self.players_score

    def get_pos(self):
        pos = np.where(self.board == 1)
        return tuple(ax[0] for ax in pos)

    def get_opponent_pos(self):
        opponent_pos = np.where(self.board == 2)
        return tuple(ax[0] for ax in opponent_pos)

    def get_player_number(self):
        return self.player_number

    # def get_maximizing_player(self):
    #     return self.maximizing_player

    # def get_time_limit(self):
    #     return self.time_limit

    # def get_time(self):`
    #     return self.time
    # TODO: Delete comments from state class


class Player(AbstractPlayer):

    def __init__(self, game_time, penalty_score):
        # keep the inheritance of the parent's (AbstractPlayer) __init__()
        AbstractPlayer.__init__(self, game_time, penalty_score)
        # initialize more fields, if needed, and the Minimax algorithm from SearchAlgos.py
        self.pos = None
        self.state = None
        # self.board = None

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        # erase the following line and implement this function.
        # self.board = board
        pos = np.where(board == 1)
        # convert pos to tuple of ints
        self.pos = tuple(ax[0] for ax in pos)
        players_score_init = [0, 0]
        self.state = State(board, players_score_init, 1)
        # TODO: maybe need to change player_number to 0?

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        start_time = time.time()
        minimax = MiniMax(self.utility_f, self.succ_f, self.perform_move_f, self.state.players_score,
                          goal=self.goal_f, heuristic_f=self.heuristic_f)

        maximizing_player = 1  # TODO: calculate according to player_number?
        depth = 1
        # time_limits = [start_time, time_limit]
        val, move = minimax.search(self.state, 5, True,
                                   self.state.players_score, start_time, time_limit)
        print(f'move is: {move}')
        assert len(self.get_indexs_by_cond(lambda x: x == 2)) == 1
        assert len(self.get_indexs_by_cond(lambda x: x == 1)) == 1
        return move
        # while True:
        #     depth += 1
        #     val, result = minimax.search(self.state, depth, True,
        #                                  self.state.players_score, start_time, time_limit)
        #     if result == "Interrupted":
        #         return move
        #     else:
        #         move = result

        # end_time = time.time()
        # if end_time - start_time > time_limit:
        #     end_time = time.time()

        # handle time limit: game will be finished, and score will be lowered

        ### making move as writen in val[1] ###

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        # erase the following line and implement this function.
        self.state.board[self.get_indexs_by_cond(lambda x: x == 2)[0]] = -1
        self.state.board[pos] = 2
        # TODO: maybe need to change

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        # erase the following line and implement this function. In case you choose not to use it, use 'pass' instead of the following line.
        current_fruits_pos = self.get_indexs_by_cond(lambda x: x > 2)
        for fruit_pos in current_fruits_pos:
            self.state.board[fruit_pos] = 0
        # if len(current_fruits_positions[0]) == 0:
        #     return
        for pos in fruits_on_board_dict.keys():
            self.state.board[pos] = fruits_on_board_dict[pos]

        assert len(self.get_indexs_by_cond(lambda x: x == 2)) == 1
        assert len(self.get_indexs_by_cond(lambda x: x == 1)) == 1

    ########## helper functions in class ##########
    # TODO: add here helper functions in class, if needed

    ########## helper functions for MiniMax algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm
    def utility_f(self, players_score):
        if players_score[0] - players_score[1] > 0:
            return 1
        elif players_score[0] - players_score[1] < 0:
            return -1
        else:
            return 0

    # returns possible directions to move
    def succ_f(self, state, pos):
        for d in self.directions:
            i = pos[0] + d[0]
            j = pos[1] + d[1]
            lines = len(state.board)
            columns = len(state.board[0])
            if 0 <= i < lines and 0 <= j < columns and \
                    (state.board[i][j] not in [-1, 1, 2]):  # then move is legal
                yield d[0], d[1]

    # gets an op and moves the player accroding to this op, prev_val will be passed before recursic call
    def perform_move_f(self, op, is_not_reversed, prev_val, players_score, move_opponent):
        player_move = 2 if move_opponent else 1
        pos = self.get_indexs_by_cond(lambda x: x == player_move)[0]
        assert len(self.get_indexs_by_cond(lambda x: x == 2)) == 1
        print(f'-move player {player_move}')
        print(f'-pos {pos}')
        if is_not_reversed:
            self.state.board[pos] = -1
            new_pos = (pos[0] + op[0], pos[1] + op[1])
            players_score[player_move - 1] += prev_val
            print(f'-put -1 in {pos}')
        else:
            assert prev_val not in [1, 2]
            self.state.board[pos[0], pos[1]] = prev_val
            players_score[player_move - 1] -= prev_val
            new_pos = (pos[0] - op[0], pos[1] - op[1])
            print(f'-put {prev_val} in {pos}')

        self.state.board[new_pos] = player_move
        self.print_board(self.state)
        print(f'-len(self.get_indexs_by_cond(lambda x: x == 2)) == {len(self.get_indexs_by_cond(lambda x: x == 2))}')
        print(f'-len(self.get_indexs_by_cond(lambda x: x == 1)) == {len(self.get_indexs_by_cond(lambda x: x == 1))}\n')
        assert len(self.get_indexs_by_cond(lambda x: x == 2)) == 1
        assert len(self.get_indexs_by_cond(lambda x: x == 1)) == 1
        return self.state

    def heuristic_f(self, state):
        closest = float('inf')
        closest_val = -1
        for fruit in self.get_indexs_by_cond(lambda x: x > 2):
            md_dist = abs(self.pos[0] - fruit[0]) + abs(self.pos[1] - fruit[1])
            if md_dist < closest:
                closest = md_dist
                closest_val = state.board[fruit]
        v1 = (closest_val / 300) if closest_val != -1 else 0
        opp_pos = self.get_indexs_by_cond(lambda x: x == 2)[0]
        option_for_op = state.state_options(opp_pos)
        v2 = 1 / option_for_op if option_for_op > 0 else 1
        option_for_me = state.state_options(self.pos)
        v3 = (1 / 3) * option_for_me
        difference = self.state.players_score[0] - self.state.players_score[1]
        v4 = difference / 150 if difference > 0 else 0
        print('########DBG PRINT########')
        self.print_board(state)
        print('#########################')

        print(f'v1={v1} , v2={v2} , v3={v3} , heuristic={(1 / 3) * (v1 + v2 + v3)}')
        return (1 / 6) * (v1 + v2 + v3) + (1/2) * v4

    def goal_f(self):
        if self.state.state_options(self.pos) == 0:
            return True
        return False

    def get_indexs_by_cond(self, cond):
        ret = []
        for i in range(len(self.state.board)):
            for j in range(len(self.state.board[0])):
                if cond(self.state.board[i][j]):
                    ret.append((i, j))
        return ret

    def print_board(self, state):
        board_to_print = np.flipud(state.board)
        print('_' * len(board_to_print[0]) * 4)
        for row in board_to_print:
            row = [str(int(x)) if x != -1 else 'X' for x in row]
            print(' | '.join(row))
            print('_' * len(row) * 4)
