import argparse
from GameWrapper import GameWrapper
import os, sys
import utils

# import matplotlib
# import numpy as np
import matplotlib.pyplot as plt


def get_winner(gameWrapper, player_index):
    if player_index and gameWrapper.some_player_cant_move:
        score_1, score_2 = gameWrapper.game.get_players_scores()
        if score_1 != score_2:
            winning_player = int(score_2 > score_1) + 1
            return winning_player
    return -1 # represents tie

if __name__ == "__main__":
    players_options = [x+'Player' for x in ['Live', 'Simple', 'Minimax', 'Alphabeta', 'GlobalTimeAB', 'LightAB',
                                            'HeavyAB', 'Compete']]

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-player1', default='MinimaxPlayer', type=str,
                        help='The type of the first player.',
                        choices=players_options)
    #SimplePlayer
    parser.add_argument('-player2', default='AlphabetaPlayer',  type=str,
                        help='The type of the second player.',
                        choices=players_options)
    
    parser.add_argument('-board', default='rectangle_board.csv', type=str,
                        help='Name of board file (.csv).')

    parser.add_argument('-move_time', default=5, type=float,
                        help='Time (sec) for each turn.')
    parser.add_argument('-game_time', default=2000, type=float,
                        help='Global game time (sec) for each player.')
    parser.add_argument('-penalty_score', default=300, type=float, 
                        help='Penalty points for a player when it cant move or its time ends.')
    parser.add_argument('-max_fruit_score', default=300, type=float, 
                        help='Max points for a fruit on board.')
    parser.add_argument('-max_fruit_time', default=15, type=float, 
                        help='Max time for fruit on the board (turns).')

    parser.add_argument('-terminal_viz', action='store_true', 
                        help='Show game in terminal only.')
    parser.add_argument('-dont_print_game', action='store_true', 
                        help='Together with "terminal_viz", show in terminal only the winner.')
    args = parser.parse_args()

    # check validity of game and turn times
    if args.game_time < args.move_time:
        raise Exception('Wrong time arguments.')

    # check validity of board file type and existance
    board_file_type = args.board.split('.')[-1]
    if board_file_type != 'csv':
        print("saar")
        raise Exception(f'Wrong board file type argument, {board_file_type}.')
    if not args.board in os.listdir('boards'):
        raise Exception(f'Board file {args.board} does not exist in "boards" directory.')

    # Players inherit from AbstractPlayer - this allows maximum flexibility and modularity
    player_1_type = 'players.' + args.player1
    player_2_type = 'players.' + args.player2
    game_time = args.game_time
    penalty_score = args.penalty_score
    __import__(player_1_type)
    __import__(player_2_type)
    player_1 = sys.modules[player_1_type].Player(game_time, penalty_score)
    player_2 = sys.modules[player_2_type].Player(game_time, penalty_score)

    board = utils.get_board_from_csv(args.board)

    # print game info to terminal
    print('Starting Game!')
    print(args.player1, 'VS', args.player2)
    print('Board', args.board)
    print('Players have', args.move_time, 'seconds to make a signle move.')
    print('Each player has', game_time, 'seconds to play in a game (global game time, sum of all moves).')

    # create game with the given args
    game = GameWrapper(board[0], board[1], board[2], player_1=player_1, player_2=player_2,
                    # terminal_viz= True,
                    terminal_viz=args.terminal_viz,
                    print_game_in_terminal=not args.dont_print_game,
                    time_to_make_a_move=args.move_time, 
                    game_time=game_time,
                    penalty_score=args.penalty_score,
                    max_fruit_score=args.max_fruit_score,
                    max_fruit_time=args.max_fruit_time)
    
    # start playing!
    game.start_game()

    for experiment in range(1, 3):
        HeavyABplayer_search_depth = 3 - (experiment - 1)
        print('Experiment'+experiment+': HeavyABplayer_search_depth is'+HeavyABplayer_search_depth)
        grades_list = []
        depth_differences_list = []
        for level_i in range(1, 4):
            lightABplayer_depth = level_i - 1 + HeavyABplayer_search_depth
            depth_differences_list.append(lightABplayer_depth-HeavyABplayer_search_depth)
            current_level_wins = 0
            for game in range(1, 6):
                player_1_type = 'players.LightABPlayer'
                player_2_type = 'players.HeavyABPlayer'
                player_1 = sys.modules[player_1_type].Player(game_time, penalty_score)
                player_2 = sys.modules[player_2_type].Player(game_time, penalty_score)

                current_game_wrapper = GameWrapper(board[0], board[1], board[2], player_1=player_1, player_2=player_2,
                                   # terminal_viz=args.terminal_viz,
                                   # print_game_in_terminal=not args.dont_print_game,
                                   terminal_viz=False,
                                   print_game_in_terminal=False,
                                   time_to_make_a_move=args.move_time,
                                   game_time=game_time,
                                   penalty_score=args.penalty_score,
                                   max_fruit_score=args.max_fruit_score,
                                   max_fruit_time=args.max_fruit_time)

                # start playing!
                current_game_wrapper.start_game()
                winner_number = get_winner(current_game_wrapper)
                current_level_wins += int(winner_number == 2)
            current_level_grade = current_level_wins / 5
            grades_list.append(current_level_grade)

        print('Experiment '+experiment+' graph:')
        # draw graph with x as depth_differences_list and y as grades_list

        # t = np.arange(0.0, 2.0, 0.01)
        # s = 1 + np.sin(2 * np.pi * t)

        # Data for plotting
        fig, ax = plt.subplots()
        ax.plot(grades_list, depth_differences_list)

        ax.set(xlabel='Depth Difference', ylabel='Grade',
               title='Grade as a function of Depth Difference')
        ax.grid()

        fig.savefig('experiment'+experiment+'.png")
        plt.show()
