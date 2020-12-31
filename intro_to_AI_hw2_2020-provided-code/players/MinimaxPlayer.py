"""
MiniMax Player
"""
import time

import numpy as np

import utils
from Game import Game
from SearchAlgos import MiniMax
from players.AbstractPlayer import AbstractPlayer
#TODO: you can import more modules, if needed


class State:
    def _init_(self, board, players_score, turn_number, maximizing_player):
        self.board = board
        self.players_score = players_score
        self.turn_number = turn_number
        # self.penalty_score = penalty_score
        self.maximizing_player = maximizing_player
        #self.time = time
        #self.time_limit = time_limit

    def state_options(self, board, pos):
        num_ops_available = 0
        for d in utils.get_directions():
            i = pos[0] + d[0]
            j = pos[1] + d[1]
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and (board[i][j] not in [-1, 1, 2]):
                num_ops_available += 1
        return num_ops_available
        # if num_ops_available == 0:
        #     return -1
        # else:
        #     return 4 - num_ops_available

    def heuristic_value(self):
        # pos = np.where(self.board == 1)
        # pos = tuple(ax[0] for ax in pos)
        # rival_pos = np.where(self.board == 2)
        # rival_pos = tuple(ax[0] for ax in rival_pos)
        # max_fruit = 0
        # if self.turn_number <= min(self.board.shape[0], self.board.shape[1]):
        #     for x in range(self.board.shape[0]):
        #         for y in range(self.board.shape[1]):
        #             if self.board[x, y] > 2:
        #                 if not self.maximizing_player:#my turn, its my succ state
        #                     md_fruit = abs(x - pos[0]) + abs(y - pos[1])
        #                 else:
        #                     md_fruit = abs(x - rival_pos[0]) + abs(y - rival_pos[1])
        #                 temp = self.board[x,y] / md_fruit
        #                 if temp > max_fruit and md_fruit < min(self.board.shape[0],self.board.shape[1])/2:
        #                     max_fruit = temp
        # md_other_player = abs(pos[0]-rival_pos[0])+abs(pos[1]-rival_pos[1])
        # stateScore=self.state_options(self.board, pos) if not self.maximizing_player else self.state_options(self.board, rival_pos)
        # return max_fruit+self.scores[0 if not self.maximizing_player else 1]-self.scores[0 if self.maximizing_player else 1]-(1/md_other_player)+stateScore+self.NR_reachable_blocks(False)-self.NR_reachable_blocks(True)+self.turn_counter

    #TODO: calc rechable in nXn square, not all board
    def reachable_white_cells(self, player_number):
        if player_number == 1:
            pos = self.get_pos()
        else:
            assert player_number==2
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
                        board_copy[i,j] = -1
                        reachable.append((i,j))
                        count += 1
                        # we limit the amount of white cells we want to check because far white cells are less relevant for us,
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

    def get_turn_number(self):
        return self.turn_number

    def get_maximizing_player(self):
        return self.maximizing_player

    # def get_time_limit(self):
    #     return self.time_limit

    # def get_time(self):`
    #     return self.time
    #TODO: Delete comments from state class

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
        #erase the following line and implement this function.
        # self.board = board
        pos = np.where(board == 1)
        # convert pos to tuple of ints
        self.pos = tuple(ax[0] for ax in pos)
        players_score_init = [0,0]
        self.state = State(board, players_score_init, 1, player_num)
        # TODO: maybe need to change turn_number to 0?

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        start_time = time.time()
        minimax = MiniMax(self.utility_f, self.succ_f(), self.perform_move_f, self.state.players_score, heuristic_f=self.heuristic_f)

        maximizing_player = 1 #TODO: calculate according to turn_number?
        depth = 1
        # time_limits = [start_time, time_limit]
        val, move, interrupted = minimax.search(self.state, depth, self.state.maximizing_player,
                                                self.state.players_score, start_time, time_limit)
        while True:
            depth+=1
            val, result, interrupted = minimax.search(self.state, depth, self.state.maximizing_player,
                                                         self.state.players_score, start_time, time_limit)
            if interrupted:
                return move
            else:
                move = result

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
        self.board[pos] = -1
        #TODO: maybe need to change


    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        # erase the following line and implement this function. In case you choose not to use it, use 'pass' instead of the following line.
        current_fruits_pos = np.where(self.board > 2)
        for fruit_pos in current_fruits_pos[0]:
            self.board[fruit_pos] = 0
        # if len(current_fruits_positions[0]) == 0:
        #     return
        for fruit_tuple in fruits_on_board_dict:
            self.board[fruit_tuple[0]] = fruit_tuple[1]


    ########## helper functions in class ##########
    #TODO: add here helper functions in class, if needed


    ########## helper functions for MiniMax algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm
    def utility_f(self, players_score):
        if players_score[0] - players_score[1] > 0:
            return 1
        elif players_score[0] - players_score[1] < 0:
            return -1
        else:
            return 0
        # return state.get_scores()[0] if maximizing_player else state.get_scores()[1]
        # diff = state.get_scores()[0] - state.get_scores()[1]
        # diff += (1 / (3 * self.board.shape[0] * self.board.shape[1])) * state.turn_counter if diff <= 0 else 0
        # if self.both_stuck(state):
        #    return diff
        # # return diff between scores, only 1 stuck
        # return diff - state.penalty_score if maximizing_player else diff + state.penalty_score


    #returns possible directions to move
    def succ_f(self):
        for d in self.directions:
            i = self.pos[0] + d[0]
            j = self.pos[1] + d[1]

            if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and \
                    (self.board[i][j] not in [-1, 1, 2]):  # then move is legal
                yield d[0], d[1]

    #gets an op and moves the player accroding to this op, prev_val will be passed before recursic call
    def perform_move_f(self, op, is_not_reversed, prev_val, players_score):
        player_number = self.board[self.pos[0], self.pos[1]]

        if is_not_reversed:
            self.board[self.pos[0], self.pos[1]] = -1
            self.pos = (self.pos[0] + op[0], self.pos[1] + op[1])
            players_score[player_number - 1] += prev_val
        else:
            self.board[self.pos[0], self.pos[1]] = prev_val
            players_score[player_number - 1] -= prev_val
            self.pos = (self.pos[0] - op[0], self.pos[1] - op[1])

        self.board[self.pos[0], self.pos[1]] = player_number

    def heuristic_f(self, ):
        pass

