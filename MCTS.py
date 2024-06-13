import numpy as np
import gym
import chess
import gym_chess
import random
import copy
import chess
from helpers import custom_reset, custom_step
from collections import defaultdict
from generatingEndgame import generate_endgame_FEN
from tqdm import tqdm

import chess.engine
engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")

env = gym.make('Chess-v0')
env.reset = custom_reset

class ChessMCTSNode():
    def __init__(self, color, state, parent=None, parent_action=None):
        self.color = color
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.num_visits = 0
        self.results = defaultdict(int)
        self.results[1] = 0
        self.results[-1] = 0
        self.untried_actions = None
        self.untried_actions = self.get_untried_actions() if not self.state._observation().is_game_over() else None
    
    def get_untried_actions(self):
        self.untried_actions = self.state.legal_moves
        return self.untried_actions

    def q_value(self):
        wins = self.results[1]
        losses = self.results[-1]
        return wins - losses

    def visit_count(self):
        return self.num_visits

    def expand(self):
        action = self.untried_actions.pop()

        next_state = copy.deepcopy(self.state)
        next_state.step(action)
        child_node = ChessMCTSNode(self.color,
            next_state, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node 

    def is_terminal_node(self):
        return self.state._observation().is_game_over()

    def rollout(self, max_moves_playout):
        current_rollout_state = copy.deepcopy(self.state)

        if current_rollout_state._observation().is_game_over():
            if current_rollout_state._observation().is_checkmate():
                if self.color:  # If the player is white
                    reward = 1
                else:
                    reward = 0
            else:
                if self.color:
                    reward = 0
                else:
                    reward = 1
        
        moves = 0
        while not current_rollout_state._observation().is_game_over() and not moves >= max_moves_playout:
            possible_moves = current_rollout_state.legal_moves
            action = self.rollout_policy(current_rollout_state, possible_moves)
            obs, reward, done, info = current_rollout_state.step(action)
            moves += 1

            if not self.color:
                reward = 1 - reward
        
        return reward

    def backpropagate(self, result):
        self.num_visits += 1.
        self.results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_param=0.1):
        choices_weights = [(c.q_value() / c.visit_count()) + exploration_param * np.sqrt((2 * np.log(self.visit_count()) / c.visit_count())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, state, possible_moves):
        return random.choice(possible_moves)

    def tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def select_best_action(self, simulations, max_moves_playout):
        for _ in range(simulations):
            selected_node = self.tree_policy()
            reward = selected_node.rollout(max_moves_playout)
            selected_node.backpropagate(reward)
        return self.best_child(exploration_param=0.1)

results = []
MCTS_simulations = 1000
max_moves_playout = 20
mate_depths = [1, 7, 11, 15]
endgames = [' KQK', ' KRK', ' KBNK']

for endgame in endgames:
    for mate_depth in mate_depths:
        path = r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess\checkmate dataset\checkmate in " + str(mate_depth) + endgame + ".txt"
        text_file = open(path)
        Endgames = text_file.read().split(',')
        print(len(Endgames))

        for Endgame in tqdm(Endgames):
            move_count = 0
            obs = env.reset(env, Endgame)
            done = False

            while not done and move_count <= 30:
                if obs.turn:
                    root = ChessMCTSNode(True, state=env)
                    selected_node = root.select_best_action(MCTS_simulations, max_moves_playout)
                    selected_move = selected_node.parent_action
                    obs, reward, done, info = env.step(selected_move)
                    move_count += 1
                else:
                    engine_move = random.choice(env.legal_moves)  # Replace with engine move logic
                    obs, reward, done, info = env.step(engine_move)

            if obs.is_checkmate():
                results.append(1)
            else:
                results.append(0)

            print(env.render())
            print('end of game')

        print(sum(results) / len(results))

env.close()
