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

chess_board = [i for i in range(64)]

chess_pieces = [chess.QUEEN]
center_squares = [(3,3),(3,4),(4,3),(4,4)]

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
random_player = False
center_index = 0


with open(r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess\Agents\KQKheuristic40ep.pickle", 'rb') as f:
    data = pickle.load(f)
    Q_table = np.array(data)


with open(r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess\Agents\KQK150kep.pickle", 'rb') as f:
    data = pickle.load(f)
    Q_table_opponent = np.array(data)

starting_positions = [generate_endgame_FEN(chess_board, chess_pieces) for i in range(nr_test)]

random_results = []


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

            if random_player:
                best_move = random.choice(moves)

            obs, reward, done, info = env.step(env, best_move)
            positions.append(obs)
            played_moves.append(best_move)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                break

        else:   
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in chess_board]
            if not moves:
                if obs.is_check():
                    checkmate = True

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
        random_results.append(0)


print(sum(random_results)/len(random_results))
print(len(random_results))

total_random_results.append(random_results)


# playing against trained agent
results = []

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

            if random_player:
                best_move = random.choice(moves)

            obs, reward, done, info = env.step(env, best_move)
            played_moves.append(best_move)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                break


        else:   
            K, Q, k = pos_to_index_KQK(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in chess_board]
            if not moves:
                if obs.is_check():
                    checkmate = True
                break

            Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table_opponent, obs, K, Q, k)
            obs, reward, done, info = env.step(env, best_move)
            played_moves.append(best_move)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                break

    if checkmate:
        results.append(1)
    else:
        results.append(0)


print(sum(results))
print(sum(results)/len(results))
print(len(results))


heuristic_results = []

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

            if random_player:
                best_move = random.choice(moves)

            obs, reward, done, info = env.step(env, best_move)
            played_moves.append(best_move)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                break

        else:   
            K, Q, k = pos_to_index_KQK(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in chess_board]

            if not moves:
                if obs.is_check():
                    checkmate = True
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
        heuristic_results.append(0)


print(sum(heuristic_results))
print(sum(heuristic_results)/len(heuristic_results))
print(len(heuristic_results))

total_heuristic_results.append(heuristic_results)



results = []

# playing against Stockfish
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

            if random_player:
                best_move = random.choice(moves)

            obs, reward, done, info = env.step(env, best_move)
            played_moves.append(best_move)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                break

        else:   
            K, Q, k = pos_to_index_KQK(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in chess_board]

            if not moves:
                if obs.is_check():
                    checkmate = True
                break

            # if sum([move for move in moves if obs.is_capture(move)]) >= 1:
            #     move_to_play = random.choice([move for move in moves if obs.is_capture(move)])

            else:

                best_move_black = engine.play(obs, chess.engine.Limit(time=0.001)).move
            
            obs, reward, done, info = env.step(env, best_move_black)
            played_moves.append(best_move_black)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                break

    if checkmate:
        results.append(1)
    else:
        results.append(0)


print(sum(results))
print(sum(results)/len(results))
print(len(results))



