import numpy as np
import gym
import chess
import gym_chess
import random
import copy
import chess
import matplotlib.pyplot as plt
from helpers import custom_reset, custom_step, find_max_Q_move, king_distance_reward_simple, king_distance_reward
from collections import defaultdict
from generatingEndgame import generate_endgame_FEN
from tqdm import tqdm

import chess.engine
engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")

env = gym.make('Chess-v0')
env.reset = custom_reset


def lookahead(env, depth):

    if depth == 0 or env._observation().is_game_over():
        return 1
    
    else:
        sum = 0
        for move in env.legal_moves:
            new_env = copy.deepcopy(env)
            new_env.step(move)
            sum += lookahead(new_env, depth-1)

        return sum
        

nr_tests = 20
depths = [depth for depth in range(5)]

average_nr_positions = []
for depth in depths:
    positions = 0
    for i in range(nr_tests):
        Fen = generate_endgame_FEN([i for i in range(64)], [chess.ROOK])
        env.reset(env, Fen)
        positions += lookahead(env, depth)
    print(positions/nr_tests)

    average_nr_positions.append(positions/nr_tests)

print('end',depths, average_nr_positions)

average_nr_positions = []
for depth in depths:
    positions = 0
    for i in range(nr_tests):
        Fen = generate_endgame_FEN([i for i in range(64)], [chess.QUEEN])
        env.reset(env, Fen)
        positions += lookahead(env, depth)
    print(positions/nr_tests)

    average_nr_positions.append(positions/nr_tests)

print('end',depths, average_nr_positions)


average_nr_positions = []
for depth in depths:
    positions = 0
    for i in range(nr_tests):
        Fen = generate_endgame_FEN([i for i in range(64)], [chess.BISHOP, chess.KNIGHT])
        env.reset(env, Fen)
        positions += lookahead(env, depth)
    print(positions/nr_tests)

    average_nr_positions.append(positions/nr_tests)

print('end',depths, average_nr_positions)


