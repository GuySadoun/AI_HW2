"""
MiniMax Player with AlphaBeta pruning
"""
import copy
import time
import numpy as np

import utils
from SearchAlgos import AlphaBeta
from players.AbstractPlayer import AbstractPlayer
from players.MinimaxPlayer import State
from players.MinimaxPlayer import pos_feasible_on_board
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
#TODO: you can import more modules, if needed


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time, penalty_score) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        #TODO: initialize more fields, if needed, and the AlphaBeta algorithm from SearchAlgos.py
        self.alpha = ALPHA_VALUE_INIT
        self.beta = BETA_VALUE_INIT
        self.pos = None
        self.state = None

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        # erase the following line and implement this function.
        pos = np.where(board == 1)
        # convert pos to tuple of ints
        self.pos = tuple(ax[0] for ax in pos)
        players_score_init = [0, 0]
        self.state = State(board, players_score_init, 1)

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        self.state.set_start_time(time.time())
        self.state.set_time_limit(time_limit)
        alphabeta = AlphaBeta(self.utility_f, self.succ_f, self.perform_move_f, self.state.players_score,
                          goal=self.goal_f, heuristic_f=self.heuristic_f)

        move = None
        minimax_val = float('-inf')
        depth = 2
        children = self.succ_f(self.state, self.pos)
        tribal_point = 1
        if len(children) == 1:
            move = children[0]
            new_pos = (self.pos[0] + move[0], self.pos[1] + move[1])
            self.state.players_score[0] += self.state.board[new_pos]
            self.state.board[self.pos] = -1
            self.state.board[new_pos] = 1
            self.pos = new_pos
            return move
        while True:
            for op in children:
                state_copy = copy.deepcopy(self.state)
                new_pos = (self.pos[0] + op[0], self.pos[1] + op[1])
                prev_val = state_copy.board[new_pos]
                assert prev_val not in [-1, -2, 1, 2]
                self.perform_move_f(state_copy, op, self.pos)
                res = alphabeta.search(state_copy, depth, True)
                if res == -2:
                    move = op if move is None else move
                    # update local board and pos
                    new_pos = (self.pos[0] + move[0], self.pos[1] + move[1])
                    self.state.players_score[0] += self.state.board[new_pos]
                    self.state.board[self.pos] = -1
                    self.state.board[new_pos] = 1
                    self.pos = new_pos
                    return move
                if res > minimax_val:
                    minimax_val = res
                    move = op
                self.perform_move_f(state_copy, op, new_pos, prev_val)
                assert len(state_copy.get_indexs_by_cond(lambda x: x == 2)) == 1
                assert len(state_copy.get_indexs_by_cond(lambda x: x == 1)) == 1
                depth += 1
            tribal_point = minimax_val / tribal_point if not np.math.isnan(minimax_val) and not np.math.isnan(
                tribal_point) else 1
            if abs(tribal_point) - 1 < 0.01:
                break

        # update local board and pos
        new_pos = (self.pos[0] + move[0], self.pos[1] + move[1]) if move is not None else self.pos
        self.state.players_score[0] += self.state.board[new_pos]
        self.state.board[self.pos] = -1
        self.state.board[new_pos] = 1
        self.pos = new_pos
        return move

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        # erase the following line and implement this function.
        self.state.board[self.state.get_indexs_by_cond(lambda x: x == 2)[0]] = -1
        self.state.players_score[1] += self.state.board[pos]
        self.state.board[pos] = 2

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        # erase the following line and implement this function. In case you choose not to use it, use 'pass' instead of the following line.
        current_fruits_pos = self.state.get_indexs_by_cond(lambda x: x > 2)
        for fruit_pos in current_fruits_pos:
            self.state.board[fruit_pos] = 0
        fruit_indexes = fruits_on_board_dict.keys()
        for pos in fruits_on_board_dict.keys():
            self.state.board[pos] = fruits_on_board_dict[pos]

        assert len(self.state.get_indexs_by_cond(lambda x: x > 2)) == len(fruit_indexes)

    ########## helper functions for MiniMax algorithm ##########
    def utility_f(self, players_score, is_my_turn):
        if is_my_turn:
            players_score[0] -= self.penalty_score
        else:
            players_score[1] -= self.penalty_score
        if players_score[0] - players_score[1] > 0:
            ret = 1
        elif players_score[0] - players_score[1] < 0:
            ret = -1
        else:
            ret = 0
        if is_my_turn:
            players_score[0] += self.penalty_score
        else:
            players_score[1] += self.penalty_score
        return ret

    # returns possible directions to move
    def succ_f(self, state, pos):
        avail_op = []
        for d in self.directions:
            i = pos[0] + d[0]
            j = pos[1] + d[1]
            lines = len(state.board)
            columns = len(state.board[0])
            if 0 <= i < lines and 0 <= j < columns and \
                    (state.board[i][j] not in [-1, 1, 2]):  # then move is legal
                avail_op.append(d)
        return avail_op

    # gets an op and moves the player accroding to this op, prev_val will be passed before recursic call
    def perform_move_f(self, state, op, pos, prev_val=-2):
        assert len(state.get_indexs_by_cond(lambda x: x == 2)) == 1
        assert len(state.get_indexs_by_cond(lambda x: x == 1)) == 1
        player_id = state.board[pos]
        if prev_val == -2:
            state.board[pos] = -1
            new_pos = (pos[0] + op[0], pos[1] + op[1])
            val_next_cell = state.board[new_pos[0]][new_pos[1]]
            state.players_score[int(player_id) - 1] += int(val_next_cell)
            state.board[new_pos] = player_id
        else:
            assert prev_val not in [1, 2]
            state.board[pos[0], pos[1]] = prev_val
            if player_id not in [1, 2]:
                print(player_id)
            state.players_score[int(player_id) - 1] -= prev_val
            last_pos = (pos[0] - op[0], pos[1] - op[1])
            state.board[last_pos] = player_id
        assert len(state.get_indexs_by_cond(lambda x: x == 2)) == 1
        assert len(state.get_indexs_by_cond(lambda x: x == 1)) == 1
        return state

    def heuristic_f(self, state, pos):
        print('***************************************************')
        closest = float('inf')
        closest_val = -1
        fruits = 0
        difference = state.players_score[0] - state.players_score[1]
        player_id = state.board[pos]
        opponent_id = player_id % 2 + 1
        opp_pos = state.get_indexs_by_cond(lambda x: x == opponent_id)[0]
        option_for_op = state.state_options(opp_pos)
        option_for_me = state.state_options(pos)
        is_opp_reachable = state.is_players_connected()
        for fruit in state.get_indexs_by_cond(lambda x: x > 2):  # find closest fruit and who's closer to max fruit
            fruits += 1
            md_dist = abs(pos[0] - fruit[0]) + abs(pos[1] - fruit[1])
            md_opp_dist = abs(opp_pos[0] - fruit[0]) + abs(opp_pos[1] - fruit[1])
            if md_dist < closest and md_dist < md_opp_dist:
                closest = md_dist
                closest_val = state.board[fruit]
                print(f'fruit at {fruit} far {closest} from {pos}')
        if fruits == 0 and difference > -self.penalty_score and is_opp_reachable:  # close your enemy strategy
            print('EAT YOUR ENEMY!')
            print(f'pos - {pos} opp_pos - {opp_pos}')
            md_from_opp = abs(pos[0] - opp_pos[0]) + abs(pos[1] - opp_pos[1])
            assert md_from_opp > 0
            reachable = self.state.reachable_white_cells(opponent_id)
            white_cells_on_board = self.state.white_cells_on_board()
            v1 = 1 / md_from_opp
            v2 = reachable / white_cells_on_board
            v3 = white_cells_on_board / self.state.board.size
            v4 = (1 / option_for_op) if option_for_op > 0 else 1
            v5 = (1 / 3) * option_for_me
            h_val = (2 / 10) * (v1 + v5) + (1 / 10) * (v2 + v3) + (4 / 10) * v4
            print(f'heuristic_f - val: {h_val}')
        elif fruits > 0:  # search fruit strategy
            print('MAX SCORE!')
            v1 = (1 / closest) * (closest_val / 300) if closest_val != -1 else 0
            v2 = 1 / option_for_op if option_for_op > 0 else 1
            option_for_me = state.state_options(self.pos)
            v3 = (1 / 3) * option_for_me
            v4 = difference / 300  # could be negative
            h_val = (1 / 6) * (v1 + v2 + v3) + (1 / 2) * v4
            print(f'heuristic_f - val 2: {h_val}')
        else:  # staying alive strategy - maximum h_val is 0.5
            print('SURVIVE!')
            reachable_for_me = self.state.reachable_white_cells(player_id)
            reachable_for_opp = self.state.reachable_white_cells(opponent_id)
            v1 = reachable_for_me / reachable_for_opp
            h_val = (1 / 2) * v1
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        return h_val

    def goal_f(self, state, pos, players_score):
        all_next_positions = [utils.tup_add(pos, direction) for direction in self.directions]
        possible_positions = [position for position in all_next_positions if pos_feasible_on_board(state, position)]
        player_cant_move = len(possible_positions) == 0

        if player_cant_move:
            player_id = state.board[pos]
            opp_id = 1 if player_id == 2 else 2
            opp_pos = state.get_indexs_by_cond(lambda x: x == opp_id)[0]
            opp_next_positions = [utils.tup_add(opp_pos, direction) for direction in self.directions]
            possible_opp_positions = [position for position in opp_next_positions if
                                      pos_feasible_on_board(state, position)]
            opp_cant_move = len(possible_opp_positions) == 0
            if not opp_cant_move:
                print('goal_f - returned True')
                return True
        print('goal_f - returned False')
        return False

    #TODO: add here the utility, succ, and perform_move functions used in AlphaBeta algorithm