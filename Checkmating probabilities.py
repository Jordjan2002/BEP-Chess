import gym
import chess
import gym_chess
import random
from helpers import custom_reset, custom_step, find_max_Q_move, pos_to_index, pos_to_index_KQK
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer
from generatingEndgame import generate_endgame_FEN

env = gym.make('Chess-v0')
env.reset = custom_reset
env.step = custom_step

pos_to_index = pos_to_index_KQK


pieces = [chess.QUEEN]

episodes = 10_000


#specifying smaller board
chess_boards = [[i for i in range(64)], [8,9,10,11,12,13,14, 16,17,18,19,20,21,22,  24,25,26,27,28,29,30, 32,33,34,35,36,37,38, 40,41,42,43,44,45,46, 48,49,50,51,52,53,54, 56,57,58,59,60,61,62], [16,17,18,19,20,21, 24,25,26,27,28,29, 32,33,34,35,36,37, 40,41,42,43,44,55, 48,49,50,51,52,53, 56,57,58,59,60,61],
[24,25,26,27,28,32,33,34,35,36,40,41,42,43,44,48,49,50,51,52,56,57,58,59,60], [33,34,35,36,40,41,42,43,48,49,50,51,56,57,58,59]] 

winning_rates = []

for chess_board in chess_boards:
    results = []
    for episode in tqdm(range(episodes)):

        played_moves = [] # keep track of moves
        #Fen = random.choice(Fens) # taking one of the FEN positions
        Fen = generate_endgame_FEN(chess_board, pieces)
        obs = env.reset(env, Fen)
        done = False
        nr_moves = 0

        while not done and len(played_moves) < 50: # with 50 move rule

            checkmate = False
            nr_moves +=1

            if obs.turn: # if White to move
                K, R, k = pos_to_index(obs.piece_map())
                moves = env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board] # only regard moves in 4x4 board

                best_move = random.choice(moves)
                obs, reward, done, info = env.step(env, best_move)

                played_moves.append(best_move)

                if done: # if the game ends new Q_value will be the reward
                    checkmate = True
                    break # go to next episode

                moves = env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board] # only regard moves in 4x4 board

                if not moves: # if black will have no moves in the smaller board after move of white it is either checkmate or stalemate and game is over
                    if obs.is_check():
                        checkmate = True
                    break


            else: # if Black to move
                K, R, k = pos_to_index(obs.piece_map())
                moves = env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board] # only regard moves in 4x4 board

                best_move = random.choice(moves)          

                obs, reward, done, info = env.step(env, best_move)

                played_moves.append(best_move)

                if done: # if the game ends new Q_value will be the reward
                    break

                

        if env._board.is_checkmate(): 
                results.append(1)

        else:
                results.append(0)


    print('winning_rate:',sum(results)/len(results))
    winning_rates.append(sum(results)/len(results))

print(winning_rates)
    
env.close()
