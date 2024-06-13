import gym
import chess
import gym_chess
import random
from helpers import custom_reset, custom_step, find_max_Q_move_KQK, pos_to_index_KQK, king_distance_reward, pos_to_index, pos_to_representation, rep_to_move
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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")

class ConvDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ConvDQN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # Fully connected layers
        self.fc3 = nn.Linear(128, 512)  # Bereken de inputgrootte op basis van de output van de laatste convolutielaag
        self.fc4 = nn.Linear(512, n_actions)


    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten de output van de laatste convolutielaag
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
n_actions = 36
# Get the number of state observations
#state, reward, done, info = env.reset()
#print(state, reward, done, info)
#state, info = env.reset()
n_observations = 4
n_observations = 192

model = ConvDQN(n_observations, n_actions)
model.load_state_dict(torch.load(r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess\model40k_13_06.pth"))
model.eval() # put model in inference mode

with open(r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess\edgames-KRK2_13_06.pickle", 'rb') as f:
    data = pickle.load(f)
    train_endgames = np.array(data)


env = gym.make('Chess-v0')
env.reset = custom_reset
env.step = custom_step

chess_pieces = [chess.ROOK]
chess_board = [i for i in range(64)]
nr_test = 5000
random_starting_positions = [generate_endgame_FEN(chess_board, chess_pieces) for i in range(nr_test)]

legals = []
results = []


for starting_pos in tqdm(np.random.choice(train_endgames, 5000)):
    obs = env.reset(env, starting_pos)
    # print(env.render())
    # print(' ')

    done = False
    while not done:
        if obs.turn:
            state = pos_to_representation(env._observation())
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            Q_values = model(state).tolist()[0]
            selected_action = np.argmax(Q_values)
            selected_move, legal = rep_to_move(selected_action,None, env)

            if not legal:
                legals.append(0)
                # print(selected_move)
                # print('Illegal move selected')

            else:
                legals.append(1)
                # print(env.render())
                # print(' ')

            obs, reward, done, info = env.step(env, selected_move)

            if done:
                if obs.is_checkmate():
                    results.append(1)
                else:
                    results.append(0)
                break

        else:
            move = random.choice(env.legal_moves)
            obs, reward, done, info = env.step(env, move)
            # print(env.render())
            # print(' ')

            if done:
                results.append(0)
                break


print(sum(legals)/len(legals))
print(len(results),sum(results)/len(results))

legal_test = []
results = []
for starting_pos in tqdm(random_starting_positions):
    obs = env.reset(env, starting_pos)
    # print(env.render())
    # print(' ')

    done = False
    while not done:
        if obs.turn:
            state = pos_to_representation(env._observation())
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            Q_values = model(state).tolist()[0]
            selected_action = np.argmax(Q_values)
            selected_move, legal = rep_to_move(selected_action,None, env)

            if not legal:
                legal_test.append(0)
                # print(selected_move)
                # print('Illegal move selected')

            else:
                legal_test.append(1)
                # print(env.render())
                # print(' ')
            obs, reward, done, info = env.step(env, selected_move)

            if done:
                if obs.is_checkmate():
                    results.append(1)
                else:
                    results.append(0)
                break

        else:
            move = random.choice(env.legal_moves)
            obs, reward, done, info = env.step(env, move)
            # print(env.render())
            # print(' ')

            if done:
                results.append(0)
                break


print(sum(legal_test)/len(legal_test))
print(len(results), sum(results)/len(results))



legal_test = []
results = []
for starting_pos in tqdm(random_starting_positions):
    obs = env.reset(env, starting_pos)
    # print(env.render())
    # print(' ')

    done = False
    while not done:
        if obs.turn:
            state = pos_to_representation(env._observation())
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            Q_values = model(state).tolist()[0]
            selected_action = np.argmax(Q_values)
            selected_move, legal = rep_to_move(selected_action,None, env)

            if not legal:
                legal_test.append(0)
                # print(selected_move)
                # print('Illegal move selected')

            else:
                legal_test.append(1)
                # print(env.render())
                # print(' ')
            obs, reward, done, info = env.step(env, selected_move)

            if done:
                if obs.is_checkmate():
                    results.append(1)
                else:
                    results.append(0)
                break

        else:
            move = engine.play(obs, chess.engine.Limit(time=0.001)).move
            obs, reward, done, info = env.step(env, move)
            # print(env.render())
            # print(' ')

            if done:
                results.append(0)
                break



print(sum(legal_test)/len(legal_test))
print(len(results), sum(results)/len(results))


for starting_pos in tqdm(np.random.choice(train_endgames, 5000)):
    obs = env.reset(env, starting_pos)
    # print(env.render())
    # print(' ')

    done = False
    while not done:
        if obs.turn:
            state = pos_to_representation(env._observation())
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            Q_values = model(state).tolist()[0]
            selected_action = np.argmax(Q_values)
            selected_move, legal = rep_to_move(selected_action,None, env)
            selected_move = random.choice(env.legal_moves)

            if not legal:
                legals.append(0)
                # print(selected_move)
                # print('Illegal move selected')

            else:
                legals.append(1)
                # print(env.render())
                # print(' ')

            obs, reward, done, info = env.step(env, selected_move)

            if done:
                if obs.is_checkmate():
                    results.append(1)
                else:
                    results.append(0)
                break

        else:
            move = engine.play(obs, chess.engine.Limit(time=0.001)).move
            obs, reward, done, info = env.step(env, move)
            # print(env.render())
            # print(' ')

            if done:
                results.append(0)
                break


print(sum(legals)/len(legals))
print(len(results),sum(results)/len(results))