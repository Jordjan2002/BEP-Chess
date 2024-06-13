import copy
import gym
import chess
import gym_chess
from helpers import custom_reset, evaluate_minimax, evaluate_minimax_KQK, king_distance_reward, evaluation_KBNK
import random
from tqdm import tqdm
import time
from generatingEndgame import generate_endgame_FEN
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")


nr_test = 50
stockfish = True # stockfish opponent, else random opponent
pieces = [chess.ROOK] # pieces to checkmate with
reward_func = king_distance_reward


def minimax(env, depth, alpha, beta, turn):

    obs = env._observation()   
    if depth == 0 or obs.is_game_over():
        # print('naar dit')
        # print(env.render())
        # print(evaluate_minimax(env))
        return king_distance_reward(env) # evaluate_old(env.observation)
    
    if turn: # if white to move:
        eval = float('-inf')
        for move in env.legal_moves:
            new_env = copy.deepcopy(env)
            new_env.step(move)
            eval = max(eval, minimax(new_env, depth-1, alpha, beta, False))
            alpha = max(alpha, eval)
            if eval > beta:
                break
        return eval
    
    else:
        eval = float('inf')
        for move in env.legal_moves:
            new_env = copy.deepcopy(env)
            new_env.step(move)
            eval = min(eval, minimax(new_env, depth-1, alpha, beta, True))
            beta = min(beta, eval)
            if eval < alpha:
                break
        return eval


def play_minimax_move(env, depth, alpha, beta, turn, positions):
    moves = env.legal_moves
    moves_score = []
    for move in moves:
        new_env = copy.deepcopy(env)
        new_env.step(move)
        
        if new_env._observation().is_repetition(): #positions.count(new_env._observation().board_fen()) >= 3: # i.e. avoid repeting positions and therefore three fold repetitions
            score = 0
            
        else:    
            score = minimax(new_env, depth, alpha, beta, not turn)

        if new_env._observation().is_checkmate():
            score = 1000

        moves_score.append((move,score))

    # print(moves_score)
    # print(max(moves_score, key=lambda x: x[1]))

    best_move, eval = max(moves_score, key=lambda x: x[1])
    #print(best_move, eval)
    return best_move



env = gym.make('Chess-v0')
env.reset = custom_reset

Endgames = [generate_endgame_FEN([i for i in range(64)], pieces) for i in range(nr_test)]


for depth in range(0,4):
    results = []
    total_moves = 0
    for Fen in tqdm(Endgames):
        nr_moves = 0
        obs = env.reset(env, Fen)
        positions = [obs.board_fen()] # used to take into account 3 fold repeitiion
        done = False

        while not done and nr_moves <= 100:
            if obs.turn: # if white to move
                best_move = play_minimax_move(copy.deepcopy(env), depth, float('-inf'), float('inf')  ,obs.turn, positions) 
                #print(nr_nodes)
                obs, reward, done, info = env.step(best_move)
                positions.append(obs.board_fen())
                nr_moves += 1

            else:
                moves = env.legal_moves

                if stockfish:
                    move = engine.play(obs, chess.engine.Limit(time=0.001)).move
                else:
                    move = random.choice(moves)
                obs, reward, done, info = env.step(move)
                positions.append(obs.board_fen())
                nr_moves += 1


        if obs.is_checkmate():
            results.append(1)
            total_moves += nr_moves
        else:
            results.append(0)


        print(env.render())
        print(' ')

    print('average nr moves to checkmate:', total_moves/sum(results))


    print('depth:',depth, 'win rate:', sum(results)/len(results))

    print('depth:',depth, 'win rate:', sum(results)/len(results))

env.close()