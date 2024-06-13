import gym
import chess
import gym_chess
import copy
import math
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from collections import namedtuple, deque
from itertools import count
from helpers import pos_to_representation, rep_to_move, move_to_rep, custom_reset, custom_step, custom_step_DQN
from stockfish import Stockfish
from tqdm import tqdm
from generatingEndgame import generate_endgame_FEN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# chess_engine = Stockfish(path="C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")


env = gym.make('Chess-v0')
env.reset = custom_reset
env.step = custom_step #custom_step

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


episodes = 100_000
BATCH_SIZE = 100
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.001
LR = 1e-3
white_epsilon = 0.9
min_epsilon = 0.05
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = (white_epsilon-min_epsilon)/(end_epsilon_decaying-start_epsilon_decaying)

# Get number of actions from gym action space
#n_actions = env.action_space.n
n_actions = 36
# Get the number of state observations
#state, reward, done, info = env.reset()
#print(state, reward, done, info)
#state, info = env.reset()
n_observations = 4
n_observations = 192

policy_net = ConvDQN(n_observations, n_actions).to(device)
target_net = ConvDQN(n_observations, n_actions).to(device)
target_net.load_state_dict(copy.deepcopy(policy_net.state_dict()))

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


#steps_done = 0
legal_moves_count = []

def select_action(state, env, random_move):
    #global steps_done
    global legal_moves_count

    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * steps_done / EPS_DECAY)
    # print(eps_threshold)
    #steps_done += 1
    if not random_move:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            Q_values = policy_net(state).tolist()[0]
            #possible_actions = [move_to_rep(move, env) for move in env.legal_moves] # all legal indices
            #Q_values_max_action = [Q_values[action] for action in possible_actions]
            # selected_action = possible_actions[np.argmax(Q_values_max_action)]
            # selected_move, legal = rep_to_move(selected_action,None, env) # 2e argument wordt momenteel ook niet gebruikt van rep_to_move

            selected_action = np.argmax(Q_values)
            selected_move, legal = rep_to_move(selected_action,None, env)

            if legal:
                legal_moves_count.append(1)
            else:
                legal_moves_count.append(0)

            return selected_move, torch.tensor([[selected_action]], device=device, dtype=torch.long), legal
    else:
        legal_moves = env.legal_moves
        selected_move = random.choice(legal_moves)
        return selected_move, torch.tensor([[move_to_rep(selected_move, env)]], device=device, dtype=torch.long), True # always returns a legal move in this case


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
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    episodes = episodes
else:
    num_episodes = 50

endgames = [generate_endgame_FEN([i for i in range(64)], [chess.ROOK]) for i in range(episodes)]
# Navigeer naar de gewenste directory
os.chdir(r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess")

with open("edgames-KRK.pickle", "wb") as f:
    pickle.dump(endgames, f)


for Fen in tqdm(endgames):
    # Initialize the environment and get its state
    obs = env.reset(env, Fen) #state = env.reset()
    state = pos_to_representation(env._observation())
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    reward_episode = 0


    for episode in count():

        if episode > 50:
            break

        if obs.turn:

            if random.random() > white_epsilon:
                move, action, legal = select_action(state, env, random_move=False)

            else:
                move, action, legal = select_action(state, env, random_move=True)

            if not legal:
                reward = -50
                reward = torch.tensor([reward], device=device)
                next_state = None
                memory.push(state, action, next_state, reward)
                reward_episode += reward
                #print(reward)
                # print(state, action, next_state, reward)
                break


            obs, reward, done, info = env.step(env, move) #observation, reward, terminated, truncated = env.step(action.item())
            reward_episode += reward
            new_state = pos_to_representation(obs) #.flatten() # hier dus weer flatten om het eerst maar eens te proberen
            reward = torch.tensor([reward], device=device)
            #done = terminated or truncated

            if done:
                next_state = None # maar de vraag of je inderdaad None moet toevoegen als het terminal state is
            else:
                # current_fen = env._observation().board_fen()
                # chess_engine.set_fen_position(current_fen)
                # best_move_black = chess_engine.get_best_move_time(1)
                # best_move_black = chess.Move.from_uci(str(best_move_black))
                best_move_black = random.choice(env.legal_moves)
                new_env = copy.deepcopy(env)
                obs_t1, reward_t1, done_t1, info_t1 = new_env.step(new_env, best_move_black)
                new_state = pos_to_representation(obs_t1) #.flatten()
                next_state = torch.tensor(new_state, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # print(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = copy.deepcopy(target_net.state_dict())
            policy_net_state_dict = copy.deepcopy(policy_net.state_dict())

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                # # if env._observation().is_checkmate:
                # #     print(env.render())
                # #     print(' ')
                # episode_durations.append(t + 1)
                # plot_durations()
                break

        else: # black just plays random moves for now
            legal_moves = env.legal_moves
            # current_fen = env._observation().board_fen()
            # chess_engine.set_fen_position(current_fen)
            # selected_move = chess_engine.get_best_move_time(1)
            # selected_move = chess.Move.from_uci(str(selected_move))
            selected_move = random.choice(legal_moves)
            obs, reward, done, info = env.step(env, selected_move)

            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                break

    white_epsilon -= epsilon_decay_value
    #print(reward_episode)
    episode_durations.append(reward_episode)


    # if i_episode % 500 == 0:        
    #     plot_durations()



# checkmates = 0
# test_games = 10


# for i in range(test_games):
#     Fen = generate_endgame_FEN([i for i in range(64)], [chess.ROOK])
#     obs = env.reset(env, Fen)
#     print(env.render())
#     print(' ')

#     if obs.turn:
#         Q_values = policy_net(state).tolist()[0]
#         selected_action = np.argmax(Q_values)
#         selected_move, legal = rep_to_move(selected_action,None, env)
#         if not legal:
#             print(selected_move)
#             print('Illegal move')
#             break
#         obs, reward, done, info = env.step(env, selected_move) #observation, reward, terminated, truncated = env.step(action.item())
#         print(env.render())
#         print(' ')

#         if done:
#             if env._observation().is_checkmate():
#                 checkmates += 1
#                 print('Checkmate!')
#             break

#     else: # black just plays random moves for now
#         # legal_moves = env.legal_moves
#         # current_fen = env._observation().board_fen()
#         # chess_engine.set_fen_position(current_fen)
#         # selected_move = chess_engine.get_best_move_time(1)
#         # selected_move = chess.Move.from_uci(str(selected_move))
#         selected_move = random.choice(legal_moves)
#         obs, reward, done, info = env.step(env, selected_move)
#         print(env.render())
#         print(' ')

#         if done:
#             print('Draw!')
#             break


# print('checkmates:', checkmates)

aggregate_legal_moves = []
select_move_count = []
for i in range(0, len(legal_moves_count), 1000):
    average_reward = sum(legal_moves_count[-i:])/len(legal_moves_count[-i:])
    aggregate_legal_moves.append(average_reward)
    select_move_count.append(i)

#print(aggregate_legal_moves)

aggregate_episode_rewards = []
for i in range(0, len(episode_durations), 100):
    average_reward = sum(episode_durations[-i:])/len(episode_durations[-i:])
    aggregate_episode_rewards.append(average_reward)

print(aggregate_episode_rewards)


with open(f"aggregate_episode_rewards_experiment.pickle", "wb") as f:
    pickle.dump(aggregate_episode_rewards, f)


plt.figure(1)
plt.plot(select_move_count, aggregate_legal_moves, label='Legal action rate')
plt.title('Rate of legal move selection')
plt.xlabel('# greedy select action')
plt.ylabel('Rate of legal moves selected')
plt.ioff()
plt.show()

plt.savefig('legalmovesexperiment')


print('Complete')
# plot_durations(show_result=True)
# plt.ioff()
# plt.show()
env.close()

torch.save(policy_net.state_dict(), r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess\model100k.pth")

