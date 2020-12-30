"""
MiniMax Player
"""
import numpy as np

from Game import Game
from SearchAlgos import MiniMax
from players.AbstractPlayer import AbstractPlayer
#TODO: you can import more modules, if needed


class Player(AbstractPlayer):

    def __init__(self, game_time, penalty_score):
        # keep the inheritance of the parent's (AbstractPlayer) __init__()
        AbstractPlayer.__init__(self, game_time, penalty_score)
        #TODO: initialize more fields, if needed, and the Minimax algorithm from SearchAlgos.py
        self.board = None
        self.pos = None


    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        #TODO: erase the following line and implement this function.
        self.board = board
        pos = np.where(board == 1)
        # convert pos to tuple of ints
        self.pos = tuple(ax[0] for ax in pos)

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        minimax = MiniMax(self.utility_f(), self.succ_f(), self.perform_move_f(), )
        val = minimax.search()
        ### making move as writen in val[1] ###


    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        #TODO: erase the following line and implement this function.
        raise NotImplementedError


    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        #TODO: erase the following line and implement this function. In case you choose not to use it, use 'pass' instead of the following line.
        raise NotImplementedError


    ########## helper functions in class ##########
    #TODO: add here helper functions in class, if needed


    ########## helper functions for MiniMax algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm
    def utility_f(self, players_score):

    #returns possible directions to move
    def succ_f(self):
        for d in self.directions:
            i = self.pos[0] + d[0]
            j = self.pos[1] + d[1]

            if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and \
                    (self.board[i][j] not in [-1, 1, 2]):  # then move is legal
                yield d[0], d[1]

    #gets an op and moves the player accroding to this op
    def perform_move_f(self, op, is_not_reversed):
        player_number = self.board[self.pos[0], self.pos[1]]

        if is_not_reversed:
            self.board[self.pos[0], self.pos[1]] = -1
            self.pos = (self.pos[0] + op[0], self.pos[1] + op[1])
        else:
            self.board[self.pos[0], self.pos[1]] = 0
            self.pos = (self.pos[0] - op[0], self.pos[1] - op[1])

        self.board[self.pos[0], self.pos[1]] = player_number


    def heuristic_f(self, ):


