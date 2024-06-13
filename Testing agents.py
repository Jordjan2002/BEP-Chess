import gym
import chess
import gym_chess
import random
from helpers import custom_reset, custom_step_reward, find_max_Q_move_KQK, pos_to_index_KQK, king_distance_reward, pos_to_index, find_max_Q_move, king_distance_reward_simple
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import chess.engine

from tqdm import tqdm
from timeit import default_timer as timer
from generatingEndgame import generate_endgame_FEN
from tempfile import TemporaryFile
from numpy import genfromtxt
from itertools import compress

engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")

#with open(r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess\Agents\4x4KQK.pickle", 'rb') as f:
# with open(r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess\qtable-KQK-avg "+board_size+".pickle", 'rb') as f:
#     data = pickle.load(f)
# Q_table = np.array(data)

chess_boards = [[32,33,34,35, 40,41,42,43 ,48,49,50,51, 56,57,58,59],[24,25,26,27,28, 32,33,34,35,36, 40,41,42,43,44, 48,49,50,51,52, 56,57,58,59,60],
                [16,17,18,19,20,21, 24,25,26,27,28,29, 32,33,34,35,36,37, 40,41,42,43,44,45, 48,49,50,51,52,53, 56,57,58,59,60,61],
                [8,9,10,11,12,13,14, 16,17,18,19,20,21,22,  24,25,26,27,28,29,30, 32,33,34,35,36,37,38, 40,41,42,43,44,45,46, 48,49,50,51,52,53,54, 56,57,58,59,60,61,62],[i for i in range(64)]] 

chess_pieces = [chess.QUEEN]
center_squares_boards = [[(1,5),(1,6),(2,5),(2,6)], [(2,5)], [(2,5),(2,6),(3,5),(3,6)], [(3,4)], [(3,3),(3,4),(4,3),(4,4)]]

env = gym.make('Chess-v0')
env.reset = custom_reset
env.step = custom_step_reward

total_random_results = []
total_random_stalemates = []
total_random_loses_piece = []
total_random_repetitions = []
total_random_fifty_move_rules = []

total_agent_results = []
total_agent_stalemates = []
total_agent_loses_piece = []
total_agent_repetitions = []
total_agent_fifty_move_rules = []

total_heuristic_results = []
total_heuristic_stalemates = []
total_heuristic_loses_piece = []
total_heuristic_repetitions = []
total_heuristic_fifty_move_rules = []

nr_test = 1000
center_index = 0

for chess_board in chess_boards:

    center_squares = center_squares_boards[center_index]

    board_size = str(int(len(chess_board)**0.5))+ "x"+ str(int(len(chess_board)**0.5))

    with open(r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess\qtable-KQK-avg "+board_size+".pickle", 'rb') as f:
        data = pickle.load(f)
        Q_table = np.array(data)

    starting_positions = [generate_endgame_FEN(chess_board, chess_pieces) for i in range(nr_test)]


    random_results = []
    random_stalemates = 0
    random_loses_piece = 0
    random_repetitions = 0
    random_fifty_move_rules = 0

    # playing against random agent
    for starting_pos in tqdm(starting_positions):
        obs = env.reset(env, starting_pos)

        done = False
        checkmate = False
        played_moves = []
        positions = []

        while not done and len(played_moves) < 100:

            if obs.turn:
                K, Q, k = pos_to_index_KQK(obs.piece_map())
                moves = env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board]

                # three_folds = []
                # for move in moves:
                #     new_env = copy.deepcopy(env)
                #     new_env.step(new_env, move)
                #     if positions.count(env._observation()) >= 3:
                #         three_folds.append(False)
                #     else:
                #         three_folds.append(True)

                # moves = list(compress(moves, three_folds))


                Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, Q, k)
                obs, reward, done, info = env.step(env, best_move)
                positions.append(obs)
                played_moves.append(best_move)
                if done:
                    if obs.is_checkmate():
                        checkmate = True
                    else:
                        if not obs.is_repetition(count=5):
                            random_stalemates += 1
                    break

            else:   
                moves = env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board]
                if not moves:
                    if obs.is_check():
                        checkmate = True
                    else:
                        random_stalemates += 1
                    break
                
                move_to_play = random.choice(moves)
                obs, reward, done, info = env.step(env, move_to_play)
                positions.append(obs)
                played_moves.append(move_to_play)
                if done:
                    if obs.is_checkmate():
                        checkmate = True
                    break

        if checkmate:
            random_results.append(1)
        else:
            if len(played_moves) >= 100:
                random_fifty_move_rules += 1
            if obs.is_insufficient_material():
                random_loses_piece += 1
            if obs.is_repetition(count=5):
                random_repetitions += 1
            random_results.append(0)


    print(sum(random_results), random_stalemates, random_fifty_move_rules, random_loses_piece, random_repetitions)
    print(sum(random_results)/len(random_results))
    print(len(random_results))

    total_random_results.append(random_results)
    total_random_stalemates.append(random_stalemates)
    total_random_loses_piece.append(random_loses_piece)
    total_random_repetitions.append(random_repetitions)
    total_random_fifty_move_rules.append(random_fifty_move_rules)

    
    # playing against trained agent
    results = []
    stalemates = 0
    fifty_move_rules = 0
    loses_piece = 0
    repetitions = 0
    for starting_pos in tqdm(starting_positions):
        obs = env.reset(env, starting_pos)

        done = False
        played_moves = []
        checkmate = False


        while not done and len(played_moves) < 100:

            if obs.turn:
                K, Q, k = pos_to_index_KQK(obs.piece_map())
                moves = env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board]
                Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, Q, k)
                obs, reward, done, info = env.step(env, best_move)
                played_moves.append(best_move)
                if done:
                    if obs.is_checkmate():
                        checkmate = True
                    else:
                        if not obs.is_repetition(count=5):
                            stalemates += 1

                    break


            else:   
                K, Q, k = pos_to_index_KQK(obs.piece_map())
                moves = env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board]
                if not moves:
                    if obs.is_check():
                        checkmate = True
                    else:
                        stalemates += 1
                    break

                Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, Q, k)
                obs, reward, done, info = env.step(env, best_move)
                played_moves.append(best_move)
                if done:
                    if obs.is_checkmate():
                        checkmate = True
                    break

        if checkmate:
            results.append(1)
        else:
            # print(env.render())
            # print(played_moves)
            if len(played_moves) >= 100:
                fifty_move_rules += 1
            if obs.is_insufficient_material():
                loses_piece += 1
            if obs.is_repetition(count=5):
                repetitions += 1

            results.append(0)


    print(sum(results), stalemates, fifty_move_rules, loses_piece, repetitions)
    print(sum(results)/len(results))
    print(len(results))
    total_agent_results.append(results)
    total_agent_stalemates.append(stalemates)
    total_agent_loses_piece.append(loses_piece)
    total_agent_repetitions.append(repetitions)
    total_agent_fifty_move_rules.append(fifty_move_rules)


    heuristic_results = []
    heuristic_stalemates = 0
    heuristic_loses_piece = 0
    heuristic_repetitions = 0
    heuristic_fifty_move_rules = 0
    # playing against heuristic agent

    #center_squares = 
    for starting_pos in tqdm(starting_positions):
        obs = env.reset(env, starting_pos)

        done = False
        played_moves = []
        checkmate = False

        while not done and len(played_moves) < 100:

            if obs.turn:
                K, Q, k = pos_to_index_KQK(obs.piece_map())
                moves = env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board]
                Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, Q, k)
                obs, reward, done, info = env.step(env, best_move)
                played_moves.append(best_move)
                if done:
                    if obs.is_checkmate():
                        checkmate = True
                    else:
                        if not obs.is_repetition(count=5):
                            heuristic_stalemates += 1
                    break

            else:   
                K, Q, k = pos_to_index_KQK(obs.piece_map())
                moves = env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board]

                if not moves:
                    if obs.is_check():
                        checkmate = True
                    else:
                        heuristic_stalemates += 1
                    break

                # if sum([move for move in moves if obs.is_capture(move)]) >= 1:
                #     move_to_play = random.choice([move for move in moves if obs.is_capture(move)])

                else:
                    move_scores = []
                    for move in moves:
                        new_env = copy.deepcopy(env)
                        new_env.step(new_env, move)
                        reward = king_distance_reward_simple(env, center_squares)
                        move_scores.append((move, reward))
                    move_to_play, score = min(move_scores,  key = lambda x: x[1])

                obs, reward, done, info = env.step(env, move_to_play)
                played_moves.append(move_to_play)
                if done:
                    if obs.is_checkmate():
                        checkmate = True
                    break

        if checkmate:
            heuristic_results.append(1)
        else:
            if len(played_moves) >= 100:
                heuristic_fifty_move_rules += 1
            if obs.is_insufficient_material():
                heuristic_loses_piece += 1
            if obs.is_repetition(count=5):
                heuristic_repetitions += 1
            heuristic_results.append(0)


    print(sum(heuristic_results), heuristic_stalemates, heuristic_fifty_move_rules, heuristic_loses_piece, heuristic_repetitions)
    print(sum(heuristic_results)/len(heuristic_results))
    print(len(heuristic_results))

    total_heuristic_results.append(heuristic_results)
    total_heuristic_stalemates.append(heuristic_stalemates)
    total_heuristic_loses_piece.append(heuristic_loses_piece)
    total_heuristic_repetitions.append(heuristic_repetitions)
    total_heuristic_fifty_move_rules.append(heuristic_fifty_move_rules)

    center_index += 1




# results = []
# stalemates = 0
# loses_piece = 0
# repetitions = 0
# fifty_move_rules = 0
# # playing against Stockfish
# for starting_pos in tqdm(starting_positions):
#     obs = env.reset(env, starting_pos)

#     done = False
#     played_moves = []
#     checkmate = False

#     while not done and len(played_moves) < 100:

#         if obs.turn:
#             K, Q, k = pos_to_index_KQK(obs.piece_map())
#             moves = env.legal_moves
#             moves = [move for move in moves if move.to_square in chess_board]
#             Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, Q, k)
#             obs, reward, done, info = env.step(env, best_move)
#             played_moves.append(best_move)
#             if done:
#                 if obs.is_checkmate():
#                     checkmate = True
#                 else:
#                     if not obs.is_repetition(count=5):
#                         stalemates += 1
#                 break

#         else:   
#             K, Q, k = pos_to_index_KQK(obs.piece_map())
#             moves = env.legal_moves
#             moves = [move for move in moves if move.to_square in chess_board]

#             if not moves:
#                 if obs.is_check():
#                     checkmate = True
#                 else:
#                     stalemates += 1
#                 break

#             # if sum([move for move in moves if obs.is_capture(move)]) >= 1:
#             #     move_to_play = random.choice([move for move in moves if obs.is_capture(move)])

#             else:

#                 best_move_black = engine.play(obs, chess.engine.Limit(time=0.1)).move
                
#             obs, reward, done, info = env.step(env, best_move_black)
#             played_moves.append(best_move_black)
#             if done:
#                 if obs.is_checkmate():
#                     checkmate = True
#                 break

#     if checkmate:
#         results.append(1)
#     else:
#         if len(played_moves) >= 100:
#             fifty_move_rules += 1
#         if obs.is_insufficient_material():
#             loses_piece += 1
#         if obs.is_repetition(count=5):
#             repetitions += 1
#         results.append(0)


# print(sum(results), stalemates, fifty_move_rules, loses_piece, repetitions)
# print(sum(results)/len(results))
# print(len(results))

random_draws = sum(total_random_stalemates+total_random_loses_piece+total_random_repetitions+total_random_fifty_move_rules)

print(sum(total_random_stalemates)/random_draws)
print(sum(total_random_loses_piece)/random_draws)
print(sum(total_random_repetitions)/random_draws)
print(sum(total_random_fifty_move_rules)/random_draws)

agent_draws = sum(total_agent_stalemates+total_agent_loses_piece+total_agent_repetitions+total_agent_fifty_move_rules)
print(sum(total_agent_stalemates)/agent_draws)
print(sum(total_agent_loses_piece)/agent_draws)
print(sum(total_agent_repetitions)/agent_draws)
print(sum(total_agent_fifty_move_rules)/agent_draws)

heuristic_draws = sum(total_heuristic_stalemates+total_heuristic_loses_piece+total_heuristic_repetitions+total_heuristic_fifty_move_rules)
print(sum(total_heuristic_stalemates)/heuristic_draws)
print(sum(total_heuristic_loses_piece)/heuristic_draws)
print(sum(total_heuristic_repetitions)/heuristic_draws)
print(sum(total_heuristic_fifty_move_rules)/heuristic_draws)

