"""
MiniMax Player
"""
import time
import numpy as np
import utils
import copy
from SearchAlgos import MiniMax
from players.AbstractPlayer import AbstractPlayer


class State:
    turn_counter = 0

    def __init__(self, board, players_score, player_number):
        self.board = board
        self.players_score = players_score
        self.player_number = player_number
        self.start_time = None
        self.time_limit = None

    def state_options(self, pos):
        num_ops_available = 0
        for d in utils.get_directions():
            i = pos[0] + d[0]
            j = pos[1] + d[1]
            if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and (self.board[i][j] not in [-1, 1, 2]):
                num_ops_available += 1
        return num_ops_available

    def white_cells_on_board(self):
        count = 0
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] not in [-1, 1, 2]:count += 1
        return count

    def reachable_white_cells(self, player_number):
        if player_number == 1:
            pos = self.get_pos()
        else:
            assert player_number == 2
            pos = self.get_opponent_pos()

        board_copy = copy.deepcopy(self.board)
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
        return count

    def is_players_connected(self):
        board_copy = copy.deepcopy(self.board)
        reachable = [self.get_pos()]
        opp_pos = self.get_opponent_pos()
        while len(reachable) > 0:
            curr_pos = reachable.pop()
            for d in utils.get_directions():
                i = curr_pos[0] + d[0]
                j = curr_pos[1] + d[1]
                if opp_pos[0] == i and opp_pos[1] == j:
                    return True
                if 0 <= i < len(board_copy) and 0 <= j < len(board_copy[0]) and (board_copy[i][j] not in [-1, 1]):
                    assert board_copy[i][j] != 2
                    board_copy[i, j] = -1
                    reachable.append((i, j))
        return False

    def get_indexs_by_cond(self, cond):
        ret = []
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if cond(self.board[i][j]):
                    ret.append((i, j))
        return ret

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

    def get_time_limit(self):
        return self.time_limit

    def get_time_left(self):
        time_passed = time.time() - self.start_time
        time_left = self.time_limit - time_passed
        return time_left

    def set_time_limit(self, time_limit):
        self.time_limit = time_limit

    def set_start_time(self, start_time):
        self.start_time = start_time

    def print_board(self):
        board_to_print = np.flipud(self.board)
        print('_' * len(board_to_print[0]) * 4)
        for row in board_to_print:
            row = [str(int(x)) if x != -1 else 'X' for x in row]
            print(' | '.join(row))
            print('_' * len(row) * 4)


def pos_feasible_on_board(state, pos):
    # on board
    on_board = (0 <= pos[0] < len(state.board) and 0 <= pos[1] < len(state.board[0]))
    if not on_board:
        return False

    # free cell
    value_in_pos = state.board[pos[0]][pos[1]]
    free_cell = (value_in_pos not in [-1, 1, 2])
    return free_cell


class Player(AbstractPlayer):

    def __init__(self, game_time, penalty_score):
        # keep the inheritance of the parent's (AbstractPlayer) __init__()
        AbstractPlayer.__init__(self, game_time, penalty_score)
        # initialize more fields, if needed, and the Minimax algorithm from SearchAlgos.py
        self.pos = None
        self.opp_pos = None
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
        opp_pos = np.where(board == 2)
        # convert pos to tuple of ints
        self.pos = tuple(ax[0] for ax in pos)
        self.opp_pos = tuple(ax[0] for ax in opp_pos)
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
        minimax = MiniMax(self.utility_f, self.succ_f, self.perform_move_f, self.state.players_score,
                          goal=self.goal_f, heuristic_f=self.heuristic_f)
        minimax_val = float('-inf')
        depth = 0
        children = self.succ_f(self.state, self.pos)
        move = children[0]
        res_for_prev_depth = 0
        tribal_point = 0
        if len(children) == 1:
            new_pos = (self.pos[0] + move[0], self.pos[1] + move[1])
            self.state.players_score[0] += self.state.board[new_pos]
            self.state.board[self.pos] = -1
            self.state.board[new_pos] = 1
            self.pos = new_pos
            return move
        while True:
            state_copy = copy.deepcopy(self.state)
            res = -1
            for op in children:
                new_pos = (self.pos[0] + op[0], self.pos[1] + op[1])
                prev_val = state_copy.board[new_pos]
                assert prev_val not in [-1, -2, 1, 2]
                self.perform_move_f(state_copy, op, self.pos)
                res = minimax.search(state_copy, depth, False, True)
                if res > minimax_val:
                    minimax_val = res
                    move = op
                self.perform_move_f(state_copy, op, new_pos, prev_val)
            if res == 0 or res_for_prev_depth == res:
                tribal_point += 1
                if tribal_point == 3:
                    break
            res_for_prev_depth = res
            depth += 1
            if state_copy.get_time_left() < 0.7:
                # print(f'move decided = {move} with val = {minimax_val}')
                break

        # update local board and pos
        self.perform_move_f(self.state, move, self.pos)
        self.pos = (self.pos[0] + move[0], self.pos[1] + move[1])
        return move

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        # erase the following line and implement this function.
        assert self.state.board[pos] not in [1, 2, -1]
        self.state.board[self.opp_pos] = -1
        self.state.players_score[1] += self.state.board[pos]
        self.state.board[pos] = 2
        self.opp_pos = pos

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
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

    # gets an op and moves the player according to this op, prev_val will be passed before recursive call
    @staticmethod
    def perform_move_f(state, op, curr_pos_on_board, prev_val=-2):
        # assert len(state.get_indexs_by_cond(lambda x: x == 2)) == 1
        # assert len(state.get_indexs_by_cond(lambda x: x == 1)) == 1
        state.turn_counter += 1 if prev_val == -2 else -1
        player_id = state.board[curr_pos_on_board]
        # assert player_id in [1, 2]
        if prev_val == -2:  # forward
            state.board[curr_pos_on_board] = -1
            new_pos = (curr_pos_on_board[0] + op[0], curr_pos_on_board[1] + op[1])
            val_next_cell = state.board[new_pos[0]][new_pos[1]]
            state.players_score[int(player_id) - 1] += int(val_next_cell)
            state.board[new_pos] = player_id
        else:               # backwards
            assert prev_val not in [1, 2]
            state.board[curr_pos_on_board] = prev_val
            last_pos = (curr_pos_on_board[0] - op[0], curr_pos_on_board[1] - op[1])
            state.players_score[int(player_id) - 1] -= prev_val
            state.board[last_pos] = player_id
        # assert len(state.get_indexs_by_cond(lambda x: x == 2)) == 1
        # assert len(state.get_indexs_by_cond(lambda x: x == 1)) == 1

    def heuristic_f(self, state):
        # print('***************************************************')
        board_len = len(state.board)
        player_id = self.state.board[self.pos]
        opponent_id = player_id % 2 + 1
        set_indexes = state.get_indexs_by_cond(lambda x: x == player_id)
        assert len(set_indexes) == 1
        pos = set_indexes[0]
        set_indexes = state.get_indexs_by_cond(lambda x: x == opponent_id)
        assert len(set_indexes) == 1
        opp_pos = set_indexes[0]
        option_for_opp = state.state_options(opp_pos)
        option_for_me = state.state_options(pos)
        is_opp_reachable_state = state.is_players_connected()
        is_opp_reachable_game = self.state.is_players_connected()
        closest_md_for_me = float('inf')
        closest_md_for_opp = float('inf')
        closest_val = -1
        fruits = 0
        max_fruit = -1
        sum_fruit = 0
        if is_opp_reachable_game and not is_opp_reachable_state and \
                abs(state.players_score[0] - state.players_score[1]) < self.penalty_score:
            reachable_for_state_opp = state.reachable_white_cells(opponent_id)
            if reachable_for_state_opp > 0:
                reachable_for_state = state.reachable_white_cells(player_id)
                if reachable_for_state - reachable_for_state_opp > 2:
                    strategy = 'SPLIT BOARD'
                    return 1

        for fruit in state.get_indexs_by_cond(lambda x: x > 2):  # find closest fruit and who's closer to max fruit
            fruits += 1
            md_dist = abs(pos[0] - fruit[0]) + abs(pos[1] - fruit[1])
            md_opp_dist = abs(opp_pos[0] - fruit[0]) + abs(opp_pos[1] - fruit[1])
            fruit_val = state.board[fruit]
            sum_fruit += state.board[fruit]
            max_fruit = max(fruit_val, max_fruit)
            if md_dist < closest_md_for_me and md_dist < board_len - state.turn_counter:
                closest_md_for_me = md_dist
                closest_val = fruit_val
            if md_opp_dist < closest_md_for_opp:
                closest_md_for_opp = md_opp_dist

        if fruits > 0 and closest_md_for_me < (board_len - state.turn_counter):  # search fruit strategy
            assert max_fruit != -1
            assert closest_md_for_opp is not np.isnan(closest_md_for_opp)
            strategy = 'MAX SCORE!'
            v1 = (1 / closest_md_for_me) * (closest_val / max_fruit)
            v2 = min(state.players_score[0] - self.state.players_score[0] / max_fruit, 1)
            v3 = min((1 / 3) * option_for_me, 1)
            h_val = (5 / 21) * v1 + (15 / 21) * v2 + (1 / 21) * v3
        else:
            reachable_for_me_for_state = state.reachable_white_cells(player_id)
            if is_opp_reachable_state:                                           # close your enemy strategy - maximum h_val is 0.8
                strategy = 'EAT YOUR ENEMY!'
                md_from_opp = abs(pos[0] - opp_pos[0]) + abs(pos[1] - opp_pos[1])
                assert md_from_opp > 0
                v1 = 1 / md_from_opp
                v2 = (1 / 3) * option_for_me
                v3 = (1 / option_for_opp) if option_for_opp > 0 else 1
                reachable_for_opp_for_state = state.reachable_white_cells(opponent_id)
                reachable_for_opp_for_state = max(reachable_for_opp_for_state, 1)  # avoid division by zero
                v4 = min(reachable_for_me_for_state / reachable_for_opp_for_state, 1)
                reachable_for_opp_on_game = self.state.reachable_white_cells(opponent_id)
                v5 = (reachable_for_opp_on_game - reachable_for_opp_for_state) / reachable_for_opp_on_game
                v6 = (1 / 3) * option_for_opp
                if reachable_for_me_for_state - reachable_for_opp_for_state > 0:  # for division move
                    v7 = 1
                else:
                    v7 = 0
                h_val = (1 / 10) * (v1 + v2 + v3 + v4 + v6) + (1 / 4) * v5 + (1 / 4) * v7
                h_val *= 0.8
            else:                                                               # staying alive strategy - maximum h_val is 0.5
                strategy = 'SURVIVE!'
                reachable_for_me_for_game = self.state.reachable_white_cells(player_id)
                assert reachable_for_me_for_game > 0
                v1 = min(reachable_for_me_for_state / reachable_for_me_for_game, 1)
                h_val = (1 / 2) * v1

        # print(f'strategy = {strategy}')
        # print(f'heuristic_f - val: {h_val}')
        # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        return h_val

    def goal_f(self, state, pos):
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
                return True
        return False
