import gym
import chess
import gym_chess
import random
from helpers import custom_reset, custom_step, find_max_Q_move, pos_to_index
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf
import keras.backend as backend
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizer_v2.adam import Adam
from keras.callbacks import TensorBoard
from collections import deque


input_shape = (8,8,3)
action_space_size_white = 36
action_space_size_black = 8

# hyperparameters
episodes = 2_000
Replay_memory_size = 10_000
MIN_REPLAY_MEMORY_SIZE = 1_000
target_network_update = 50 
Mini_batch_size = 64
discount = 0.99
start_epsilon_decaying = 1
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

MODEL_NAME_White = "DQNchess white"
MODEL_NAME_Black = "DQNchess black"
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20
Show_every = 50

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()


ep_rewards_white = [-200]
ep_rewards_black = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
#tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


class chessagentDQN():

    def __init__(self, model_name, input_shape, action_space_size):
        self.input_shape = input_shape
        self.action_space_size = action_space_size

        self.model = self.create() # policy network

        self.target_model = self.create() # target network
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=Replay_memory_size)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logstest/{model_name}-{int(time.time())}")
        self.target_update_counter = 0


    def create(self):
        model = Sequential()

        model.add(Conv2D(192, (3, 3), input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Conv2D(100, (3, 3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Conv2D(50, (3, 3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Flatten()) 
        model.add(Dense(self.action_space_size))

        model.add(Dense(self.action_space_size, activation='linear')) 
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        return model
    
    def update_replay_memory(self, experience):
        self.replay_memory.append(experience)

    def get_Qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]    
    
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, Mini_batch_size)

        current_states = np.array([experience[0] for experience in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.asarray([experience[3] for experience in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_Q = np.max(future_qs_list[index])
                new_q = reward + discount*max_future_Q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            Y.append(current_qs)

        self.model.fit(np.array(X), np.array(Y), batch_size=Mini_batch_size, verbose=0
                       , shuffle=False, callbacks=[self.tensorboard] if terminal_state else None) # Only update tensorboard when game is finished
        
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > target_network_update:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


def pos_to_representation(position): 
    "preprocesses chess position for CNN with one-hot-encoding"
    pieces_pos = position.piece_map()
    pieces_pos = {v: k for k, v in pieces_pos.items()}

    if position.is_game_over():
        R = False
        K, k = pieces_pos[chess.Piece.from_symbol('K')], pieces_pos[chess.Piece.from_symbol('k')]
    else:
        K, R, k = pieces_pos[chess.Piece.from_symbol('K')], pieces_pos[chess.Piece.from_symbol('R')], pieces_pos[chess.Piece.from_symbol('k')]

    white_king = np.zeros((8,8))
    white_rook = np.zeros((8,8))
    black_king = np.zeros((8,8))

    white_king[chess.square_rank(K)][chess.square_file(K)] = 1
    black_king[chess.square_rank(k)][chess.square_file(k)] = 1

    if R:
        white_rook[chess.square_rank(R)][chess.square_file(R)] = 1

    return np.array([white_king, white_rook, black_king]).reshape((8,8,3))


def rep_to_move(repr:int, current_position, env):
    "converts prediction of neural network to move, where the order is King, Rook"

    position = env._observation()
    position_map = position.piece_map()
    pieces_pos = {v: k for k, v in position_map.items()}

    if repr <= 7:
        if position.turn:
            piece = chess.Piece.from_symbol('K')
        else:
            piece = chess.Piece.from_symbol('k')

        king_position = pieces_pos[piece]
        rank, file = chess.square_rank(king_position),chess.square_file(king_position)
        directions = {0:[-1,1],1:[0,1],2:[1,1],3:[1,0],4:[1,-1],5:[0,-1],6:[-1,-1],7:[-1,0]}
        d_file, d_rank = directions[repr]
        new_rank, new_file = rank+d_rank, file+d_file
        new_square = chess.square(new_file, new_rank)

        if 0 <= new_square <= 62: # check if new_square is on the board
            move = chess.Move(king_position, new_square)
            if move in env.legal_moves:
                return move, True

    elif 7 < repr <= 35:
        piece = chess.Piece.from_symbol('R')
        rook_position = pieces_pos[piece]
        rank, file = chess.square_rank(rook_position),chess.square_file(rook_position)

        if repr <= 14:
            d_file, d_rank = 0, repr-7
        elif 15 <= repr <= 21:
            d_file, d_rank = repr-14, 0
        elif 22 <= repr <= 28:
            d_file, d_rank = 0, 21-repr
        elif 29 <= repr <= 35:
            d_file, d_rank = 28-repr, 0

        new_rank, new_file = rank+d_rank, file+d_file
        new_square = chess.square(new_file, new_rank)
        
        if 0 <= new_square <= 62: # check if new_square is on the board
            move = chess.Move(rook_position, new_square)
            if move in env.legal_moves: # so if move within the board and legal
                return move, True
    
    return random.choice(env.legal_moves), False # the move is not legal, so return random move and False


def move_to_rep(move, env):
    old_square = move.from_square
    new_square = move.to_square

    position = env._observation()
    position = position.piece_map()
    
    piece = position[old_square]
    directions = {(-1,1):0,(0,1):1,(1,1):2,(1,0):3,(1,-1):4,(0,-1):5,(-1,-1):6,(-1,0):7}

    if piece == chess.Piece.from_symbol('K') or piece == chess.Piece.from_symbol('k'):
        difference = (chess.square_file(new_square) - chess.square_file(old_square), chess.square_rank(new_square) - chess.square_rank(old_square))
        return directions[difference]
    
    else:
        d_file, d_rank = [chess.square_file(new_square) - chess.square_file(old_square), chess.square_rank(new_square) - chess.square_rank(old_square)]
        
        if d_file == 0:
            if d_rank > 0:
                return d_rank
            else:
                return -d_rank + 21
        else:
            if d_file > 0:
                return d_file + 14
            else:
                return -d_file + 28


Chessnetwork_white = chessagentDQN(MODEL_NAME_White, input_shape, action_space_size_white)
Chessnetwork_black = chessagentDQN(MODEL_NAME_Black, input_shape, action_space_size_black)

env = gym.make('Chess-v0')
env.reset = custom_reset
env.step = custom_step

Fens = ['2R5/1K6/8/1k6/8/8/8/8 w - - 0 1', '8/2R5/1k1K4/8/8/8/8/8 w - - 0 1', '1k6/8/8/RK6/8/8/8/8 w - - 0 1', '8/K1k5/1R6/8/8/8/8/8 w - - 0 1',
        '8/8/3k4/RK6/8/8/8/8 w - - 0 1', '8/1k6/8/1KR5/8/8/8/8 w - - 0 1']

starting_state = random.choice(Fens) # 1.initialize random starting position
obs = env.reset(env, starting_state)
print(pos_to_representation(obs))
print(env.render())
print(rep_to_move(25, obs, env))


for episode in tqdm(range(1, episodes + 1), ascii=True, unit='episodes'):

    starting_state = random.choice(Fens) # 1.initialize random starting position
    obs = env.reset(env, starting_state)
    episode_reward_white = 0
    episode_reward_black = 0
    step_white = 1
    step_black = 1

    done = False
    moves_amount = 0
    while not done and moves_amount <= 30:

        moves_amount += 1

        if obs.turn: # if white to move EVEN CHECKEN OF DIT UBERHAUPT WEL VERANDERD!!!

            current_state = pos_to_representation(env._observation()) # preprocess to matrix representation
            legal_moves = env.legal_moves

            if not np.random.random() > epsilon: # 1. Select action via exploration or exploitation
                
                selected_move = random.choice(legal_moves)
                selected_action = move_to_rep(selected_move, env)
                move_legal = True

            else:
                all_actions = np.argsort(Chessnetwork_white.get_Qs(current_state))

                for i in range(action_space_size_white):
                    move, legal = rep_to_move(all_actions[i], obs, env)
                    if legal:
                        selected_action = all_actions[i]
                        selected_move = move
                        break

            
            new_state, reward, done, info = env.step(env, selected_move) # 2. execute selected move and 3. observe reward and next state
            new_state = pos_to_representation(new_state)
            episode_reward_white += reward
            step_white += 1

            if done:
                Chessnetwork_white.update_replay_memory((current_state, selected_action, reward, new_state, done))
                Chessnetwork_white.train(done, step_white)
                break

            all_actions = np.argsort(Chessnetwork_black.get_Qs(current_state))

            for i in range(action_space_size_black):
                move, legal = rep_to_move(all_actions[i], obs, env)
                if legal:
                    best_move_black = all_actions[i]
                    selected_move = move
                    break

            new_env = copy.deepcopy(env)
            new_state_white, reward_t1, done_t1, info = new_env.step(new_env, selected_move)
            new_state_white = pos_to_representation(new_state_white)

            Chessnetwork_white.update_replay_memory((current_state, selected_action, reward, new_state_white, done)) # 4. store experience in replay memory
            Chessnetwork_white.train(done, step_white)

            current_state = new_state

        else:

            current_state = pos_to_representation(env._observation()) # preprocess to matrix representation
            legal_moves = env.legal_moves

            if not np.random.random() > epsilon: # 1. Select action via exploration or exploitation
                
                selected_move = random.choice(legal_moves)
                selected_action = move_to_rep(selected_move, env)
                move_legal = True

            else:
                all_actions = np.argsort(Chessnetwork_black.get_Qs(current_state))

                for i in range(action_space_size_black):
                    move, legal = rep_to_move(all_actions[i], obs, env)
                    if legal:
                        selected_action = all_actions[i]
                        selected_move = move
                        break

            new_state, reward, done, info = env.step(env, selected_move) # 2. execute selected move and 3. observe reward and next state
            episode_reward_black += reward
            new_state = pos_to_representation(new_state)

            if done:
                Chessnetwork_black.update_replay_memory((current_state, selected_action, reward, new_state, done))
                Chessnetwork_black.train(done, step_black)
                break

            for i in range(action_space_size_white):
                move, legal = rep_to_move(all_actions[i], obs, env)
                if legal:
                    best_move_white = all_actions[i]
                    selected_move = move
                    break
            
            new_env = env.deepcopy(env)
            new_state_black, reward_t1, done_t1, info = new_env.step(new_env, selected_move)
            new_state_black = pos_to_representation(new_state_black)

            Chessnetwork_black.update_replay_memory((current_state, selected_action, reward, new_state_black, done)) # 4. store experience in replay memory
            Chessnetwork_black.train(done, step_black)
                        
            current_state = new_state

    ep_rewards_white.append(episode_reward_white)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards_white[-AGGREGATE_STATS_EVERY:])/len(ep_rewards_white[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards_white[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards_white[-AGGREGATE_STATS_EVERY:])
        Chessnetwork_white.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        if min_reward >= MIN_REWARD:
            Chessnetwork_white.model.save(f'models/{MODEL_NAME_White}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    ep_rewards_black.append(episode_reward_black)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards_black[-AGGREGATE_STATS_EVERY:])/len(ep_rewards_black[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards_black[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards_black[-AGGREGATE_STATS_EVERY:])
        Chessnetwork_black.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        if min_reward >= MIN_REWARD:
            Chessnetwork_black.model.save(f'models/{MODEL_NAME_Black}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    if episode % Show_every == 0:
        env.render()

env.close()



