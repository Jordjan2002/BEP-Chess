import gym
import chess
import chess.engine
import gym_chess
import random
from helpers import custom_reset, custom_step, find_max_Q_move, pos_to_index, find_max_Q_move_KQK, king_distance_reward, pos_to_index_KQK, king_distance_reward_simple
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import time
from tqdm import tqdm
from timeit import default_timer as timer
from generatingEndgame import generate_endgame_FEN
from stockfish import Stockfish

env = gym.make('Chess-v0')
env.reset = custom_reset
env.step = custom_step
pos_to_index = pos_to_index_KQK 

#chess_engine = Stockfish(path="C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")
import chess.engine
engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")


nr_of_pieces_white = 2
nr_of_pieces_black = 1
squares = 64
pieces = [chess.QUEEN]

# initiating Q-learning
learning_rate = 0.01
Show_Every = 500
discount = 0.99
episodes = 100_000
white_epsilon = 0.9 #0.5
black_epsilon = 0.2
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = white_epsilon/(end_epsilon_decaying-start_epsilon_decaying)

Q_table_white = np.zeros([squares, squares, squares,nr_of_pieces_white,squares])
Q_table_black = np.ones([squares, squares, squares,nr_of_pieces_white,squares])*50
Q_table = np.append(Q_table_white, Q_table_black, axis=3)

squares_small_board = [i for i in range(64)]

results = []
white_epsilon_values = []
black_epsilon_values = []
aggr_results = {'ep':[], 'avg':[], 'min': [], 'max': [], 'white epsilon':[], 'black epsilon':[] }


for episode in tqdm(range(episodes)):

    played_moves = [] # keep track of moves
    #Fen = random.choice(Fens) # taking one of the FEN positions
    Fen = generate_endgame_FEN(squares_small_board, pieces)
    obs = env.reset(env, Fen)
    done = False
    nr_moves = 0

    while not done and len(played_moves) < 100: # with 50 move rule

        checkmate = False
        nr_moves +=1

        if obs.turn: # if White to move
            K, R, k = pos_to_index(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in squares_small_board] # only regard moves in 4x4 board

            if np.random.random() > white_epsilon: # chance based on epsilon to look into random move
                Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, R, k)
            else:
                Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, R, k, random_move=True)
            obs, reward, done, info = env.step(env, best_move)

            #played_moves.append(best_move)

            if done: # if the game ends new Q_value will be the reward
                Q_table[K][R][k][a][b] = reward
                if obs.is_checkmate():
                    checkmate = True
                break # go to next episode


            # look one extra step ahead because of different Q updating function.
            new_env = copy.deepcopy(env)
            # current_fen = new_env._observation().board_fen()
            # best_move_black = engine.play(new_env._observation(), chess.engine.Limit(time=0.0001)).move
            #best_move_black = chess.Move.from_uci(str(best_move_black))
            K_t1, R_t1, k_t1 = pos_to_index_KQK(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in squares_small_board] # only regard moves in 4x4 board

            if not moves: # if black will have no moves in the smaller board after move of white it is either checkmate or stalemate and game is over
                if obs.is_check():
                    checkmate = True
                    reward = 50
                    Q_table[K][R][k][a][b] = reward
                else:
                    reward = -50
                    Q_table[K][R][k][a][b] = reward
                break

            move_scores = []
            for move in moves:
                copy_env = copy.deepcopy(env)
                copy_env.step(copy_env, move)
                reward = king_distance_reward_simple(copy_env)
                move_scores.append((move, reward))
            move_to_play, score = min(move_scores,  key = lambda x: x[1])

            #Q_value_best_move_t1, best_move_t1, a_t1, b_t1 = find_max_Q_move_KQK(moves, Q_table, obs, K_t1, R_t1, k_t1)
            obs_t1, reward_t1, done_t1, info_t1 = new_env.step(new_env, move_to_play)
            

            # find max Q-value after opponent would play best move
            if done_t1: # if black can end the game
                est_fut_value = -50
                Q_table[K][R][k][a][b] = Q_value_best_move + learning_rate*(reward+discount*est_fut_value-Q_value_best_move)

            else:
                K_t2, R_t2, k_t2 = pos_to_index(obs.piece_map())
                moves = new_env.legal_moves
                moves = [move for move in moves if move.to_square in squares_small_board] # only regard moves in 4x4 board

                Q_value_best_move_t2, best_move_t2, a_t2, b_t2 = find_max_Q_move_KQK(moves, Q_table, obs, K_t2, R_t2, k_t2)

                #update the value in the Q-table
                Q_table[K][R][k][a][b] =  Q_value_best_move + learning_rate*(reward+discount*Q_value_best_move_t2-Q_value_best_move) 


        else: # if Black to move

            K, Q, k = pos_to_index(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in squares_small_board]

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
                    reward = king_distance_reward_simple(env)
                    move_scores.append((move, reward))
                move_to_play, score = min(move_scores,  key = lambda x: x[1])

            obs, reward, done, info = env.step(env, move_to_play)
            #played_moves.append(move_to_play)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                break           


            # Train against Stockfish
            # current_fen = env._observation().board_fen()
            # #chess_engine.set_fen_position(current_fen)
            # best_move_black = engine.play(obs, chess.engine.Limit(time=0.1)).move
            # #best_move_black = chess.Move.from_uci(str(best_move_black))

            # obs, reward, done, info = env.step(env, best_move_black)
            # played_moves.append(best_move_black)
            

    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        white_epsilon -= epsilon_decay_value

    if env._board.is_checkmate(): 
        results.append(1)

    else:   
        results.append(0)

    white_epsilon_values.append(white_epsilon)
    black_epsilon_values.append(black_epsilon)


# code for printing out the results
    if episode % Show_Every == 0:
        print('episode:', episode)
        if env._board.is_checkmate(): # or reward == -1: # last two to make sure it prints it out well if checkmate with 4x4 board
            result = 'checkmate'
            #results.append(1)
        else:
            result = 'draw'
            #results.append(0)
        print(result, 'in', nr_moves, 'moves')  
        print('reward:', reward)
        #print(f'Episode: {episode} avg: {average_reward} min: {min(results[-Show_Every:])} max: {max(results[-Show_Every:])}')    
        print(' ')   

    if not episode % Show_Every:
        average_reward = sum(results[-Show_Every:])/len(results[-Show_Every:])
        print(average_reward)
        aggr_results['ep'].append(episode)
        aggr_results['avg'].append(average_reward)
        aggr_results['min'].append(min(results[-Show_Every:]))
        aggr_results['max'].append(max(results[-Show_Every:]))
        aggr_results['white epsilon'].append(white_epsilon)
        aggr_results['black epsilon'].append(black_epsilon)

os.chdir(r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess")

# Opslaan van het object naar een pickle-bestand in de huidige directory

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(Q_table, f)

    
env.close()

plt.plot(aggr_results['ep'], aggr_results['avg'], label='avg win rate')
# plt.plot(aggr_results['ep'], aggr_results['min'], label='min win rate')
# plt.plot(aggr_results['ep'], aggr_results['max'], label='max win rate')
#plt.plot(aggr_results['ep'], aggr_results['white epsilon'], label='white epsilon')
# plt.plot(aggr_results['ep'], aggr_results['black epsilon'], label='black epsilon')
plt.xlabel('episode')
plt.ylabel('value')
plt.legend()
plt.show()



