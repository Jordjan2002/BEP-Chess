import gym
import chess
import gym_chess
import random
from helpers import custom_reset, custom_step_reward, find_max_Q_move_KBNK, pos_to_index_KBNK
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer
from generatingEndgame import generate_endgame_FEN
from tempfile import TemporaryFile

env = gym.make('Chess-v0')
env.reset = custom_reset
env.step = custom_step_reward

nr_of_pieces_white = 3
nr_of_pieces_black = 1
squares = 64

# initiating Q-learning
learning_rate = 0.01
Show_Every = 500
discount = 0.99
episodes = 150_000
white_epsilon = 0.9 #0.5
black_epsilon = 0.9
min_epsilon = 0.05
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = (white_epsilon-min_epsilon)/(end_epsilon_decaying-start_epsilon_decaying)

Q_table_white = np.zeros([squares, squares, squares,squares,nr_of_pieces_white,squares])
Q_table_black = np.ones([squares, squares, squares,squares,nr_of_pieces_black,squares])*50
Q_table = np.append(Q_table_white, Q_table_black, axis=3)

results = []
white_epsilon_values = []
black_epsilon_values = []
aggr_results = {'ep':[], 'avg':[]}


board_index = 0

for episode in tqdm(range(episodes+1)):

    played_moves = [] # keep track of moves
    #Fen = random.choice(Fens) # taking one of the FEN positions
    Fen = generate_endgame_FEN([i for i in range(64)], [chess.BISHOP, chess.KNIGHT])
    obs = env.reset(env, Fen)
    done = False
    nr_moves = 0

    while not done and len(played_moves) < 100: # with 50 move rule

        checkmate = False
        nr_moves +=1

        if obs.turn: # if White to move
            K, B, N, k = pos_to_index_KBNK(obs.piece_map())
            moves = env.legal_moves


            if np.random.random() > white_epsilon: # chance based on epsilon to look into random move
                Q_value_best_move, best_move, a, b = find_max_Q_move_KBNK(moves, Q_table, obs, K, B, N, k)
            else:
                Q_value_best_move, best_move, a, b = find_max_Q_move_KBNK(moves, Q_table, obs, K, B, N, k, random_move=True)
            obs, reward, done, info = env.step(env, best_move)

            played_moves.append(best_move)

            if done: # if the game ends new Q_value will be the reward
                Q_table[K][B][N][k][a][b] = reward
                if obs.is_checkmate():
                    checkmate = True
                break # go to next episode


            # look one extra step ahead because of different Q updating function.
            new_env = copy.deepcopy(env)
            K_t1, B_t1, N_t1, k_t1 = pos_to_index_KBNK(obs.piece_map())
            moves = new_env.legal_moves

            Q_value_best_move_t1, best_move_t1, a_t1, b_t1 = find_max_Q_move_KBNK(moves, Q_table, obs, K_t1, B_t1, N_t1, k_t1) 
            obs_t1, reward_t1, done_t1, info_t1 = new_env.step(new_env, best_move_t1)
            

            # find max Q-value after opponent would play best move
            if done_t1: # if black can end the game
                est_fut_value = -100
                Q_table[K][B][N][k][a][b] = Q_value_best_move + learning_rate*(reward+discount*est_fut_value-Q_value_best_move)

            else:
                K_t2, B_t2, N_t2, k_t2 = pos_to_index_KBNK(obs.piece_map())
                moves = new_env.legal_moves

                Q_value_best_move_t2, best_move_t2, a_t2, b_t2 = find_max_Q_move_KBNK(moves, Q_table, obs, K_t2, B_t2, N_t2, k_t2)

                #update the value in the Q-table
                Q_table[K][B][N][k][a][b] =  Q_value_best_move + learning_rate*(reward+discount*Q_value_best_move_t2-Q_value_best_move) 


        else: # if Black to move
            K, B,N, k = pos_to_index_KBNK(obs.piece_map())
            moves = env.legal_moves

            if np.random.random() > black_epsilon: # chance based on epsilon to look into random move
                Q_value_best_move, best_move, a, b = find_max_Q_move_KBNK(moves, Q_table, obs, K, B,N, k)
            else:
                Q_value_best_move, best_move, a, b = find_max_Q_move_KBNK(moves, Q_table, obs, K, B,N, k, random_move=True)            

            obs, reward, done, info = env.step(env, best_move)

            played_moves.append(best_move)

            if done: # if the game ends new Q_value will be the reward
                Q_table[K][B][N][k][a][b] = reward # 1-reward, because interest of black is opposite
                break


            # look one extra step ahead because of different Q updating function.
            new_env = copy.deepcopy(env)
            K_t1, R_t1, k_t1 = pos_to_index_KBNK(obs.piece_map())
            moves = new_env.legal_moves

            Q_value_best_move_t1, best_move_t1, a_t1, a_t2 = find_max_Q_move_KBNK(moves, Q_table, obs, K_t1, R_t1, k_t1)
            obs_t1, reward_t1, done_t1, info_t1 = new_env.step(new_env, best_move_t1)

            if done_t1:
                #Q_table[K][R][k][a][b] = -reward_t1
                est_fut_value = 0 #reward_t1
                Q_table[K][B][N][k][a][b] =  Q_value_best_move + learning_rate*(reward+discount*est_fut_value-Q_value_best_move) # ipv het gewoon (-) reward maken


            # find max Q-value after opponent would play best move
            else:
                K_t2, R_t2, k_t2 = pos_to_index_KBNK(obs.piece_map())
                moves = new_env.legal_moves

                if not moves: # if black will have no moves in the smaller board after best move of white it is either checkmate or stalemate and game is over
                    if obs_t1.is_check():
                        estimated_future_value = 0
                        Q_table[K][B][N][k][a][b] = Q_value_best_move + learning_rate*(reward+discount*estimated_future_value-Q_value_best_move)
                    else:
                        estimated_future_value = 0
                        Q_table[K][B][N][k][a][b] = Q_value_best_move + learning_rate*(reward+discount*estimated_future_value-Q_value_best_move)

                else:
                    Q_value_best_move_t2, best_move_t2, a_t2, b_t2 = find_max_Q_move_KBNK(moves, Q_table, obs, K_t2, R_t2, k_t2)
                    Q_table[K][B][N][k][a][b] =  Q_value_best_move + learning_rate*(reward+discount*Q_value_best_move_t2-Q_value_best_move)
        

    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        white_epsilon -= epsilon_decay_value
        black_epsilon -= epsilon_decay_value

    if env._board.is_checkmate(): #or reward == -1: # last two to make sure it prints it out well if checkmate with 4x4 board
            results.append(1)

    else: # not env._board.is_checkmate() or reward == 50: # or reward == -1: # last two to make sure it prints it out well if checkmate with 4x4 board
            results.append(0)

        # white_epsilon_values.append(white_epsilon)
        # black_epsilon_values.append(black_epsilon)


# code for printing out the results
    if episode % Show_Every == 0:
            print('episode:', episode)
            print(env.render())
            if env._board.is_checkmate(): # or reward == -1: # last two to make sure it prints it out well if checkmate with 4x4 board
                result = 'checkmate'
                #results.append(1)
            else:
                result = 'draw'
                #results.append(0)
            print(result, 'in', nr_moves, 'moves')  
            print(played_moves)
            print('reward:', reward)
            #print(f'Episode: {episode} avg: {average_reward} min: {min(results[-Show_Every:])} max: {max(results[-Show_Every:])}')    
            print(' ')   

    if not episode % Show_Every:
        average_reward = sum(results[-Show_Every:])/len(results[-Show_Every:])
        aggr_results['avg'].append(average_reward)
        aggr_results['ep'].append(episode)



# saving the Q-table

outfile = TemporaryFile()
np.save(outfile, Q_table)
    
env.close()

plt.plot(aggr_results['ep'], aggr_results['avg'], label='avg win rate')
plt.title('Q-learning King Queen vs king on full board')
plt.xlabel('episode')
plt.ylabel('value')
plt.legend()
plt.show()

