import numpy as np
import gym
import chess
import gym_chess
import random
import copy
import chess
from helpers import custom_reset, custom_step, find_max_Q_move, king_distance_reward_simple, king_distance_reward
from collections import defaultdict
from generatingEndgame import generate_endgame_FEN
from tqdm import tqdm

import chess.engine
engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")

env = gym.make('Chess-v0')
env.reset = custom_reset

class MonteCarloTreeSearchNode():
    def __init__(self, colour, state, parent=None, parent_action=None):
        self.colour = colour
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions() if not self.state._observation().is_game_over() else None
        return
    
    def untried_actions(self):
        #print(self.state.render())
        #print(' ')
        self._untried_actions = self.state.legal_moves
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        
        action = self._untried_actions.pop()

        next_state = copy.deepcopy(self.state)
        next_state.step(action)
        child_node = MonteCarloTreeSearchNode(self.colour,
            next_state, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node 


    def is_terminal_node(self):
        return self.state._observation().is_game_over()


    def rollout(self, max_move_playout):
        current_rollout_state = copy.deepcopy(self.state)

        # game could already be finished here!
        if current_rollout_state._observation().is_game_over():
            if current_rollout_state._observation().is_checkmate():
                if self.colour: # if the player is white
                    reward = 1
                else:
                    reward = 0
            else:
                if self.colour:
                    reward = 0
                else:
                    reward = 1

        
        moves = 0
        while not current_rollout_state._observation().is_game_over() and not moves >= max_move_playout:
            
            possible_moves = current_rollout_state.legal_moves
            
            action = self.rollout_policy(current_rollout_state, possible_moves)

            #current_rollout_state = copy.deepcopy(current_rollout_state)
            obs, reward, done, info = current_rollout_state.step(action) #current_rollout_state = current_rollout_state.step(action)
            moves += 1

            if not self.colour:
                reward = 1 - reward

        
        return reward


    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)


    def is_fully_expanded(self):
        return len(self._untried_actions) == 0


    def best_child(self, c_param=0.1):
        
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]


    def rollout_policy(self, env, possible_moves):

        # if env._observation().turn:

        #     moves_score = []
        #     for move in possible_moves:
        #         new_env = copy.deepcopy(env)
        #         obs, reward, done, info  = new_env.step(move)
        #         score = king_distance_reward_simple(new_env)
        #         moves_score.append((move,score))
            
        #     move, eval = min(moves_score, key=lambda x: x[1])

        #     #move = engine.play(env._observation(), chess.engine.Limit(time=0.1)).move

        # else:
        #     for play in env.legal_moves:
        #         new_env = copy.deepcopy(env)
        #         new_env.step(play)

        #         if new_env._observation().is_checkmate():
        #             return play
                
        #     move = random.choice(env.legal_moves)

        # return move
    
        return random.choice(possible_moves)
    
        #return engine.play(env._observation(), chess.engine.Limit(time=0.001)).move


    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node


    def best_action(self, simulations, max_move_playout):
        
        for i in range(simulations):
            
            v = self._tree_policy()
            reward = v.rollout(max_move_playout)
            v.backpropagate(reward)
        
        return self.best_child(c_param=0.1)
    

# def main():
#     root = MonteCarloTreeSearchNode(state = env)
#     selected_node = root.best_action()
#     return selected_node.parent_action


# print(main())

# print(' ')
# print(env.render())

results = []
MCTS_simulations = 1000
max_move_playout = 20
nr_endgames = 10
Endgames = [generate_endgame_FEN([i for i in range(64)], [chess.ROOK]) for i in range(nr_endgames)]

path = r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess\checkmate dataset\checkmate in 11 KBNK.txt"
text_file = open(path)
Endgames = text_file.read().split(',')
print(len(Endgames))
Endgames = ['8/8/8/5k2/8/1K6/1Q6/8 w - - 0 1', '8/8/4k3/8/8/8/2K5/1Q6 w - - 0 1','8/8/7Q/5k2/8/8/1K6/8 w - - 0 1','3K4/1Q6/8/8/8/8/3k4/8 w - - 0 1','5Q2/5K2/8/8/8/2k5/8/8 w - - 0 1',
            '6Q1/6K1/8/8/8/3k4/8/8 w - - 0 1','8/8/8/2Q5/5k2/8/K7/8 w - - 0 1','3Q4/2K5/8/8/8/8/4k3/8 w - - 0 1','1Q1K4/8/8/8/8/5k2/8/8 w - - 0 1','8/1QK5/8/8/5k2/8/8/8 w - - 0 1',
            '8/8/3k4/8/Q7/8/8/2K5 w - - 0 1','8/8/8/8/1k6/6K1/7Q/8 w - - 0 1','5K2/8/8/8/8/2k5/5Q2/8 w - - 0 1','2K5/8/8/8/8/2k5/8/5Q2 w - - 0 1','8/1K6/8/1Q6/8/4k3/8/8 w - - 0 1',
            '7Q/3K4/8/8/8/3k4/8/8 w - - 0 1','2K1Q3/8/8/8/8/8/2k5/8 w - - 0 1','1Q6/3K4/8/8/8/3k4/8/8 w - - 0 1','8/8/Q7/1K3k2/8/8/8/8 w - - 0 1','8/1QK5/8/8/8/4k3/8/8 w - - 0 1']

# print(Endgames)
# Endgames = [chess.Board(fen=fen).mirror().fen() for fen in Endgames]
#print([chess.Board(fen=fen).mirror().board_fen() for fen in Endgames])

#endgames_nr = 20
#Endgames = [generate_endgame_FEN([i for i in range(64)], [chess.BISHOP, chess.KNIGHT]) for i in range(endgames_nr)]

for Endgame in tqdm(Endgames):

    move_count = 0

    #Endgame = '8/2k5/3R4/2K5/8/8/8/8 b - - 0 1'
    obs = env.reset(env, Endgame)
    # print(env.render())
    # print(' ')

    done = False
    while not done and move_count <= 30:

        if obs.turn:
            root = MonteCarloTreeSearchNode(True, state = env)
            selected_node = root.best_action(MCTS_simulations, max_move_playout)
            selected_move = selected_node.parent_action
            obs, reward, done, info = env.step(selected_move)
            move_count += 1

        else:

            # root = MonteCarloTreeSearchNode(False, state = env)
            # selected_node = root.best_action()
            # selected_move = selected_node.parent_action

            engine_move = engine.play(obs, chess.engine.Limit(time=0.1)).move
            engine_move = random.choice(env.legal_moves)
            obs, reward, done, info = env.step(engine_move)


    if obs.is_checkmate():
        results.append(1)
    else:
        results.append(0)

        # moves = env.legal_moves
        # selected_move = random.choice(moves)
        # obs, reward, done, info = env.step(selected_move)
        # print(env.render())
        # print(' ')
    print(env.render())
    print('end of game')

print(sum(results)/ len(results))

env.close()