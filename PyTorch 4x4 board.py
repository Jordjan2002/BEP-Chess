import gym
import chess
import chess.syzygy
import chess.engine
import gym_chess
import copy
import math
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import namedtuple, deque
from itertools import count
from helpers import custom_reset, legal_moves_smaller_board, custom_step_DQN, evaluate_pos, evaluate, pos_to_representation_smaller_board, rep_to_move_smaller_board, move_to_rep_smaller_board, custom_step_DQN_smaller_board
from generatingEndgame import generate_endgame_FEN
from stockfish import Stockfish
from tqdm import tqdm
from torchsummary import summary

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

chess_engine = Stockfish(path="C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")
engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")

env = gym.make('Chess-v0')
env.reset = custom_reset
env.step = custom_step_DQN_smaller_board #custom_step
Fen = '8/8/8/8/8/1k6/8/1KR5 w - - 0 1'

obs = env.reset(env, Fen) 
print(env.render())

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

class ConvDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ConvDQN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3)

        # Fully connected layers
        self.fc3 = nn.Linear(256, 124)
        self.fc4 = nn.Linear(124, n_actions)


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

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
episodes = 20_000

BATCH_SIZE = 100
GAMMA = 0.99
epsilon = 0.9
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = epsilon/(end_epsilon_decaying-start_epsilon_decaying)
min_epsilon = 0.05
TAU = 0.01
LR = 1e-3

# Get number of actions from gym action space
#n_actions = env.action_space.n
n_actions = 20
# Get the number of state observations
#state, reward, done, info = env.reset()
#print(state, reward, done, info)
#state, info = env.reset()
n_observations = 16

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
#print(summary(policy_net, input_size=128))

steps_done = 0

squares = [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]

def select_action(state, env, epsilon=epsilon):
    global steps_done
    sample = random.random()
    # eps_threshold = max(epsilon, min_epsilon)
    # epsilon -= epsilon_decay_value
    steps_done += 1
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            Q_values = policy_net(state).tolist()[0]
            legal_actions = [move_to_rep_smaller_board(move, env) for move in legal_moves_smaller_board(squares, env)] # all legal indices
            Q_values_max_action = [Q_values[action] for action in legal_actions]
            selected_action = legal_actions[np.argmax(Q_values_max_action)]
            selected_move, legal = rep_to_move_smaller_board(selected_action,None, env) # 2e argument wordt momenteel ook niet gebruikt van rep_to_move
            return selected_move, torch.tensor([[selected_action]], device=device, dtype=torch.long)
    else:
        selected_move = random.choice(legal_moves_smaller_board(squares, env))
        return selected_move, torch.tensor([[move_to_rep_smaller_board(selected_move, env)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards sum')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = episodes
else:
    num_episodes = 50

checkmates = []
aggr_results = {'ep':[], 'avg':[], 'min': [], 'max': [], 'epsilon':[]}



for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and get its state
    obs = env.reset(env, Fen) #state = env.reset()
    state = pos_to_representation_smaller_board(env._observation())
    state = state #state.flatten()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    reward_episode = 0

    for t in count():

        # if t > 50:
        #     break

        if obs.turn:
            move, action = select_action(state, env, epsilon)
            #reward_before = engine.analyse(board=obs, limit=chess.engine.Limit(depth=5))['score'].relative.score(mate_score=10000)
            reward_before = evaluate(env)
            obs, reward, done, info = env.step(env, move) #observation, reward, terminated, truncated = env.step(action.item())

            #done = terminated or truncated

            if done or not legal_moves_smaller_board(squares, env):
                next_state = None # maar de vraag of je inderdaad None moet toevoegen als het terminal state is -> denk het wel
            else: 
                # this would be the best move of the engine for whole board
                # current_fen = env._observation().board_fen()
                # chess_engine.set_fen_position(current_fen)
                # best_move_black = chess_engine.get_best_move_time(1)
                # best_move_black = chess.Move.from_uci(str(best_move_black))

                possible_moves_black = legal_moves_smaller_board(squares, env)
                possible_captures = [move for move in possible_moves_black if obs.is_capture(move)]
                if len(possible_captures) >= 1: # if black can capture the rook, then it does so
                    move_black = random.choice(possible_captures)
                else:
                    move_black = random.choice(possible_moves_black)
                obs, reward_t1, done, info_t1 = env.step(env, move_black)

                # if done or not legal_moves_smaller_board(squares, env): # if black ends the game, e.g. captures the rook
                #     reward = -100

                new_state = pos_to_representation_smaller_board(obs) #.flatten()

                if done or not legal_moves_smaller_board(squares, env): # if black ends the game, e.g. captures the rook
                    reward = -100
                    next_state = None

                next_state = torch.tensor(new_state, dtype=torch.float32, device=device).unsqueeze(0)
                #reward_after = engine.analyse(board=obs, limit=chess.engine.Limit(depth=5))['score'].relative.score(mate_score=10000)
                # if not done:
                #     reward_after = evaluate(env)
                #     #reward = reward_after/100 - reward_before/100 - 0.01 # reward gets overwritten if next state is not a terminal state
                #     reward = reward_after - reward_before

            reward_episode += reward
            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done or not legal_moves_smaller_board(squares, env):
                # # if env._observation().is_checkmate:
                # #     print(env.render())
                # #     print(' ')
                # episode_durations.append(t + 1)
                # plot_durations()
                break

        else: # only if in initial position black is to move
            legal_moves = legal_moves_smaller_board(squares, env)
            current_fen = env._observation().board_fen()
            chess_engine.set_fen_position(current_fen)
            selected_move = chess_engine.get_best_move_time(1)
            selected_move = chess.Move.from_uci(str(selected_move))
            #selected_move = random.choice(legal_moves)
            obs, reward, done, info = env.step(env, selected_move)
            state = pos_to_representation_smaller_board(env._observation())
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                break

        epsilon -= epsilon_decay_value

    episode_durations.append(reward_episode)
    
    if obs.is_checkmate():
        checkmates.append(1)
    else:
        checkmates.append(0)

    # if i_episode % 500 == 0:        
    #     plot_durations()

        average_reward = sum(checkmates[-500:])/len(checkmates[-500:])
        aggr_results['ep'].append(i_episode)
        aggr_results['avg'].append(average_reward)
        aggr_results['min'].append(min(checkmates[-500:]))
        aggr_results['max'].append(max(checkmates[-500:]))
        aggr_results['epsilon'].append(epsilon)

plt.plot(aggr_results['ep'], aggr_results['avg'], label='avg win rate')
plt.plot(aggr_results['ep'], aggr_results['min'], label='min win rate')
plt.plot(aggr_results['ep'], aggr_results['max'], label='max win rate')
plt.plot(aggr_results['ep'], aggr_results['epsilon'], label='epsilon')
plt.xlabel('episode')
plt.ylabel('value')
plt.legend()
plt.show()





checkmates = 0
test_games = 50


# for i in range(test_games):
#     Fen = generate_endgame_FEN()
#     obs = env.reset(env, Fen)
#     if obs.turn:
#         Q_values = policy_net(state).tolist()[0]
#         legal_actions = [move_to_rep(move, env) for move in env.legal_moves] # all legal indices
#         Q_values_max_action = [Q_values[action] for action in legal_actions]
#         selected_action = legal_actions[np.argmax(Q_values_max_action)]
#         selected_move, legal = rep_to_move(selected_action,None, env)
#         obs, reward, done, info = env.step(env, selected_move) #observation, reward, terminated, truncated = env.step(action.item())

#         if done:
#             if env._observation().is_checkmate():
#                 checkmates += 1
#             break

#     else: # black just plays random moves for now
#         legal_moves = env.legal_moves
#         current_fen = env._observation().board_fen()
#         chess_engine.set_fen_position(current_fen)
#         selected_move = chess_engine.get_best_move_time(1)
#         selected_move = chess.Move.from_uci(str(selected_move))
#         #selected_move = random.choice(legal_moves)
#         obs, reward, done, info = env.step(env, selected_move)

#         if done:
#             break


print('checkmates:', checkmates)


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
env.close()


env.close()