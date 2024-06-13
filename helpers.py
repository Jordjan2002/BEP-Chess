import chess
import random
import numpy as np
import copy


def custom_reset(self, Fen):
    'creates the position of the given Fen, so can be used to create a custom position'
    self._board = chess.Board(Fen)
    self._ready = True
    return self._observation()


def custom_step(self, action: chess.Move):


    if action not in self._board.legal_moves and not chess.Move.null():
        raise ValueError(
            f"Illegal move {action} for board position {self._board.fen()}"
        )

    self._board.push(action)

    observation = self._observation()
    #reward = self._reward()

########################################################

    obs = self._observation() # board position
    if obs.turn: # then black made the last move
        if obs.is_insufficient_material(): # if black has captured the rook
            reward = 50
        else:
            reward = 1 # reward of 1 for making a move

    else: # if white made the last move
        if obs.is_checkmate():
            reward = 50
        elif obs.is_stalemate():
            reward = -50
        # elif obs.is_repetition(count=2):
        #     reward = -20
        else:
            reward = -1 #-1

#######################################################
    done = self._board.is_game_over()

    if done:
        self._ready = False

    return observation, reward, done, None


def custom_step_reward(self, action: chess.Move):

    if action not in self._board.legal_moves and not chess.Move.null():
        raise ValueError(
            f"Illegal move {action} for board position {self._board.fen()}"
        )

    self._board.push(action)

    observation = self._observation()
    reward = self._reward() # deze dus aanpassen

########################################################

    obs = self._observation() # board position
    if obs.turn: # then black made the last move
        if obs.is_insufficient_material(): # if black has captured the rook
            reward = 100
        else:
            reward = 1 # reward of 1 for making a move

    else: # if white made the last move
        if obs.is_checkmate():
            reward = 100
        elif obs.is_stalemate():
            reward = -100
        else:
            reward = -1

#######################################################
    done = self._board.is_game_over()

    if done:
        self._ready = False

    return observation, reward, done, None


def move_to_index(move, obs):
    pieces = {chess.Piece.from_symbol('K'):0, chess.Piece.from_symbol('R'):1, chess.Piece.from_symbol('k'):2} # Q or R
    dest = move.to_square
    piece = obs.piece_map()[move.from_square]
    return [pieces[piece], dest] #pos_dic[dest]


def pos_to_index(position): # converts position to indices of KRk for Q-tables
    #pos_dic = {32:0,33:1,34:2,35:3,40:4,41:5,42:6,43:7,48:8,49:9,50:10,51:11,56:12,57:13,58:14,59:15}
    pos = {v: k for k, v in position.items()}
    state = [pos[chess.Piece.from_symbol('K')], pos[chess.Piece.from_symbol('R')], pos[chess.Piece.from_symbol('k')]] # Q or R
    return [x for x in state] #pos_dic[x]


def find_max_Q_move(moves, Q_table, obs, K, R, k, random_move=False):   
    moves_Q_values = []
    for move in moves:
        a,b = move_to_index(move, obs)
        move_Q_value = Q_table[K][R][k][a][b]
        moves_Q_values.append((move_Q_value,move, a, b))

    if not random_move:
        Q_value_best_move, best_move, a, b = max(moves_Q_values, key=lambda x: x[0])

    else:
        Q_value_best_move, best_move, a, b = random.choice(moves_Q_values)

    return Q_value_best_move, best_move, a, b


def move_to_index_KQK(move, obs):
    pieces = {chess.Piece.from_symbol('K'):0, chess.Piece.from_symbol('Q'):1, chess.Piece.from_symbol('k'):2} # Q or R
    dest = move.to_square
    piece = obs.piece_map()[move.from_square]
    return [pieces[piece], dest] #pos_dic[dest]


def move_to_index_KBNK(move, obs):
    pieces = {chess.Piece.from_symbol('K'):0, chess.Piece.from_symbol('B'):1, chess.Piece.from_symbol('N'):2, chess.Piece.from_symbol('k'):3} # Q or R
    dest = move.to_square
    piece = obs.piece_map()[move.from_square]
    return [pieces[piece], dest] #pos_dic[dest]


def pos_to_index_KBNK(position): # converts position to indices of KRk for Q-tables
    #pos_dic = {32:0,33:1,34:2,35:3,40:4,41:5,42:6,43:7,48:8,49:9,50:10,51:11,56:12,57:13,58:14,59:15}
    pos = {v: k for k, v in position.items()}
    state = [pos[chess.Piece.from_symbol('K')], pos[chess.Piece.from_symbol('B')], pos[chess.Piece.from_symbol('N')], pos[chess.Piece.from_symbol('k')]] # Q or R
    return [x for x in state] #pos_dic[x]


def pos_to_index_KQK(position): # converts position to indices of KRk for Q-tables
    #pos_dic = {32:0,33:1,34:2,35:3,40:4,41:5,42:6,43:7,48:8,49:9,50:10,51:11,56:12,57:13,58:14,59:15}
    pos = {v: k for k, v in position.items()}
    state = [pos[chess.Piece.from_symbol('K')], pos[chess.Piece.from_symbol('Q')], pos[chess.Piece.from_symbol('k')]] # Q or R
    return [x for x in state] #pos_dic[x]


def find_max_Q_move_KQK(moves, Q_table, obs, K, R, k, random_move=False):   
    moves_Q_values = []
    for move in moves:
        a,b = move_to_index_KQK(move, obs)
        move_Q_value = Q_table[K][R][k][a][b]
        moves_Q_values.append((move_Q_value,move, a, b))

    if not random_move:
        Q_value_best_move, best_move, a, b = max(moves_Q_values, key=lambda x: x[0])

    else:
        Q_value_best_move, best_move, a, b = random.choice(moves_Q_values)

    return Q_value_best_move, best_move, a, b


def find_max_Q_move_KBNK(moves, Q_table, obs, K, B, N, k, random_move=False):   
    moves_Q_values = []
    for move in moves:
        a,b = move_to_index_KBNK(move, obs)
        move_Q_value = Q_table[K][B][N][k][a][b]
        moves_Q_values.append((move_Q_value,move, a, b))

    if not random_move:
        Q_value_best_move, best_move, a, b = max(moves_Q_values, key=lambda x: x[0])

    else:
        Q_value_best_move, best_move, a, b = random.choice(moves_Q_values)

    return Q_value_best_move, best_move, a, b


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

    return np.array([white_king, white_rook, black_king]).reshape((3,8,8)) # dit was eers 8,8,3!


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
                return d_rank + 7
            else:
                return -d_rank + 21
        else:
            if d_file > 0:
                return d_file + 14
            else:
                return -d_file + 28
            

def custom_step_DQN(self, action: chess.Move):

    if action not in self._board.legal_moves:
        raise ValueError(
            f"Illegal move {action} for board position {self._board.fen()}"
        )

    self._board.push(action)

    observation = self._observation()
    reward = self._reward() # deze dus aanpassen

########################################################

    obs = self._observation() # board position
    if obs.turn: # then black made the last move
        print('ERROR')
        if obs.is_insufficient_material(): # if black has captured the rook
            reward = 40
        else:
            reward = 1 # reward of 1 for making a move

    else: # if white made the last move
        if obs.is_checkmate():
            reward = 1000
        elif obs.is_stalemate():
            reward = -1000
        # else:
            
        #     # moves_black = self.legal_moves
        #     # if sum([obs.is_capture(move) for move in moves_black]) >= 1: # if black can legally capture the rook
        #     #     reward = -1000

        else:
            reward = -1
            # position = obs.piece_map()
            # pos = {v: k for k, v in position.items()}
            # white_king_square = pos[chess.Piece.from_symbol('K')]
            # black_king_square = pos[chess.Piece.from_symbol('k')]
            # reward = (8-len(moves_black)) + (7-chess.square_distance(black_king_square, white_king_square))


#######################################################
    done = self._board.is_game_over()

    if done:
        self._ready = False

    return observation, reward, done, None



def evaluate_pos(env):
    obs = copy.deepcopy(env._observation())
    obs.push(chess.Move.null())
    moves_black = obs.legal_moves.count()
    position = obs.piece_map()
    pos = {v: k for k, v in position.items()}
    white_king_square = pos[chess.Piece.from_symbol('K')]
    black_king_square = pos[chess.Piece.from_symbol('k')]

    black_king_file = chess.square_file(black_king_square)
    black_king_rank = chess.square_rank(black_king_square)

    distance_back_rank = min([black_king_file, black_king_rank, 7-black_king_file, 7-black_king_rank])

    return (8-moves_black) + (7-chess.square_distance(black_king_square, white_king_square)) - distance_back_rank # breathing space king, distance kings, minus distance to backrank


def evaluate(env):
    obs = copy.deepcopy(env._observation())
    obs.push(chess.Move.null())
    moves_black = obs.legal_moves.count()
    position = obs.piece_map()
    pos = {v: k for k, v in position.items()}
    white_king_square = pos[chess.Piece.from_symbol('K')]
    black_king_square = pos[chess.Piece.from_symbol('k')]
    rook_square = pos[chess.Piece.from_symbol('R')]

    black_king_file = chess.square_file(black_king_square)
    black_king_rank = chess.square_rank(black_king_square)

    white_king_file = chess.square_file(white_king_square)
    white_king_rank = chess.square_rank(white_king_square)

    rook_file = chess.square_file(rook_square)
    rook_rank = chess.square_rank(rook_square)

    distance_back_rank = min([black_king_file, black_king_rank, 7-black_king_file, 7-black_king_rank])

    #calculating box-size
    if black_king_file >= rook_file and black_king_rank > rook_rank: #Q1
        box_size = (7-rook_file)*(7-rook_rank)
    elif black_king_file >= rook_file and black_king_rank <= rook_rank:
        box_size = (7-rook_file)*(rook_rank)
    elif black_king_file < rook_file and black_king_rank <= rook_rank:
        box_size = (rook_file)*(rook_rank)
    elif black_king_file < rook_file and black_king_rank > rook_rank:
        box_size = (rook_file)*(7-rook_rank)

    manhattan_distance_kings = abs(black_king_file - white_king_file) + abs(black_king_rank-white_king_rank)


    return (14-manhattan_distance_kings)*4 + (49-box_size) - chess.square_distance(white_king_square, black_king_square) # breathing space king, distance kings, minus distance to backrank


def evaluate_minimax(env):
    
    obs = env._observation()

    if obs.is_checkmate():
        return 100
    elif obs.is_stalemate():
        return 0
    elif obs.is_insufficient_material():
        return 0
    else:
        obs = copy.deepcopy(obs)
        obs.push(chess.Move.null())
        moves_black = obs.legal_moves.count()
        position = obs.piece_map()
        pos = {v: k for k, v in position.items()}
        white_king_square = pos[chess.Piece.from_symbol('K')]
        black_king_square = pos[chess.Piece.from_symbol('k')]
        rook_square = pos[chess.Piece.from_symbol('R')]

        black_king_file = chess.square_file(black_king_square)
        black_king_rank = chess.square_rank(black_king_square)

        white_king_file = chess.square_file(white_king_square)
        white_king_rank = chess.square_rank(white_king_square)

        rook_file = chess.square_file(rook_square)
        rook_rank = chess.square_rank(rook_square)

        distance_back_rank = min([black_king_file, black_king_rank, 7-black_king_file, 7-black_king_rank])

        #calculating box-size
        if black_king_file >= rook_file and black_king_rank > rook_rank: #Q1
            box_size = (7-rook_file)*(7-rook_rank)
        elif black_king_file >= rook_file and black_king_rank <= rook_rank:
            box_size = (7-rook_file)*(rook_rank)
        elif black_king_file < rook_file and black_king_rank <= rook_rank:
            box_size = (rook_file)*(rook_rank)
        elif black_king_file < rook_file and black_king_rank > rook_rank:
            box_size = (rook_file)*(7-rook_rank)

        manhattan_distance_kings = abs(black_king_file - white_king_file) + abs(black_king_rank-white_king_rank)


        return 49-box_size - manhattan_distance_kings #(14-manhattan_distance_kings)*4 + (49-box_size) - distance_back_rank*2 # breathing space king, distance kings, minus distance to backrank chess.square_distance(white_king_square, rook_square)


def evaluate_minimax_KQK(env):
    
    obs = env._observation()

    if obs.is_checkmate():
        return 100
    elif obs.is_stalemate():
        return 0
    elif obs.is_insufficient_material():
        return 0
    else:

        if obs.turn:
            obs = copy.deepcopy(obs)
            obs.push(chess.Move.null())
            moves_black = obs.legal_moves.count()

        else:
            moves_black = obs.legal_moves.count()

        position = obs.piece_map()
        pos = {v: k for k, v in position.items()}
        white_king_square = pos[chess.Piece.from_symbol('K')]
        black_king_square = pos[chess.Piece.from_symbol('k')]
        rook_square = pos[chess.Piece.from_symbol('Q')]

        black_king_file = chess.square_file(black_king_square)
        black_king_rank = chess.square_rank(black_king_square)

        white_king_file = chess.square_file(white_king_square)
        white_king_rank = chess.square_rank(white_king_square)

        rook_file = chess.square_file(rook_square)
        rook_rank = chess.square_rank(rook_square)

        distance_back_rank = min([black_king_file, black_king_rank, 7-black_king_file, 7-black_king_rank])

        #calculating box-size
        if black_king_file >= rook_file and black_king_rank > rook_rank: #Q1
            box_size = (7-rook_file)*(7-rook_rank)
        elif black_king_file >= rook_file and black_king_rank <= rook_rank:
            box_size = (7-rook_file)*(rook_rank)
        elif black_king_file < rook_file and black_king_rank <= rook_rank:
            box_size = (rook_file)*(rook_rank)
        elif black_king_file < rook_file and black_king_rank > rook_rank:
            box_size = (rook_file)*(7-rook_rank)

        manhattan_distance_kings = abs(black_king_file - white_king_file) + abs(black_king_rank-white_king_rank)

        return 25-moves_black #(14-manhattan_distance_kings)*4 + (49-box_size) - distance_back_rank*2 # breathing space king, distance kings, minus distance to backrank chess.square_distance
    
def king_distance_reward(env):
    
    obs = env._observation()

    if obs.is_checkmate():
        return 100
    elif obs.is_stalemate():
        return 0
    elif obs.is_insufficient_material():
        return 0
    else:

        position = obs.piece_map()
        pos = {v: k for k, v in position.items()}
        white_king_square = pos[chess.Piece.from_symbol('K')]
        black_king_square = pos[chess.Piece.from_symbol('k')]

        black_king_file = chess.square_file(black_king_square)
        black_king_rank = chess.square_rank(black_king_square)

        white_king_file = chess.square_file(white_king_square)
        white_king_rank = chess.square_rank(white_king_square)

        center_squares = [(3,3),(3,4),(4,3),(4,4)]
        man_dist_center = min([abs(black_king_file - file) + abs(black_king_rank-rank) for file,rank in center_squares])

        manhattan_distance_kings = abs(black_king_file - white_king_file) + abs(black_king_rank-white_king_rank)

        return 4.7*man_dist_center + 1.6*(14-manhattan_distance_kings)
    

def king_distance_reward_simple(env, center_squares = [(3,3),(3,4),(4,3),(4,4)]):
    
    obs = env._observation()

    if obs.is_checkmate():
        return 100
    elif obs.is_stalemate():
        return 0
    elif obs.is_insufficient_material():
        return 0
    else:

        position = obs.piece_map()
        pos = {v: k for k, v in position.items()}
        black_king_square = pos[chess.Piece.from_symbol('k')]

        black_king_file = chess.square_file(black_king_square)
        black_king_rank = chess.square_rank(black_king_square)

        center_squares = [(3,3),(3,4),(4,3),(4,4)]
        man_dist_center = min([abs(black_king_file - file) + abs(black_king_rank-rank) for file,rank in center_squares])

        return man_dist_center
    

def evaluation_KBNK(env):
    
    obs = env._observation()

    if obs.is_checkmate():
        return 1000
    elif obs.is_stalemate():
        return 0
    elif obs.is_insufficient_material():
        return 0
    else:

        if obs.turn:
            new_obs = copy.deepcopy(obs)
            new_obs.push(chess.Move.null())
            moves_black = obs.legal_moves.count()

        else:
            moves_black = obs.legal_moves.count()


        position = obs.piece_map()
        pos = {v: k for k, v in position.items()}
        white_king_square = pos[chess.Piece.from_symbol('K')]
        white_bishop_square = pos[chess.Piece.from_symbol('B')]
        black_king_square = pos[chess.Piece.from_symbol('k')]


        if white_bishop_square in  chess.SquareSet(chess.BB_LIGHT_SQUARES):
            corner_squares = [(0,7), (7,0)]
        else:
            corner_squares = [(0,0), (7,7)]


        black_king_file = chess.square_file(black_king_square)
        black_king_rank = chess.square_rank(black_king_square)

        white_king_file = chess.square_file(white_king_square)
        white_king_rank = chess.square_rank(white_king_square)

        center_squares = [(4,4),(4,5),(5,4),(5,5)]
        man_dist_center = min([abs(black_king_file - file) + abs(black_king_rank-rank) for file,rank in center_squares])

        man_dist_corner = min([abs(black_king_file - file) + abs(black_king_rank-rank) for file,rank in corner_squares])

        manhattan_distance_kings = abs(black_king_file - white_king_file) + abs(black_king_rank-white_king_rank)

        return man_dist_corner #10*man_dist_corner + 1.6*(14-manhattan_distance_kings) + 2*man_dist_center + 2*(8-moves_black)


def legal_moves_smaller_board(squares, env):
    all_legal_moves = env.legal_moves
    return [move for move in all_legal_moves if move.to_square in squares]


def is_checkmate_smaller_board(squares, obs):

    if len(legal_moves_smaller_board(squares, obs)) == 0 and obs.is_check():
        return True
    else:
        return False


def is_stalemate_smaller_board(squares, obs):

    if len(legal_moves_smaller_board(squares, obs)) == 0 and not obs.is_check():
        return True
    else:
        return False


def custom_step_DQN_smaller_board(self, action: chess.Move):

    if action not in self._board.legal_moves:
        raise ValueError(
            f"Illegal move {action} for board position {self._board.fen()}"
        )

    self._board.push(action)

    observation = self._observation()
    reward = self._reward() # deze dus aanpassen

########################################################

    squares = [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27] # 4x4 board of bottom left corner

    obs = self._observation() # board position
    if obs.turn: # then black made the last move
        if obs.is_insufficient_material(): # if black has captured the rook
            reward = 40
        else:
            reward = 1 # reward of 1 for making a move

    else: # if white made the last move
        if obs.is_checkmate() or is_checkmate_smaller_board(squares, obs):
            reward = 100
        elif obs.is_stalemate() or is_stalemate_smaller_board(squares, obs):
            reward = -100
        else:
            reward = -1
 


#######################################################
    done = self._board.is_game_over()

    if done:
        self._ready = False

    return observation, reward, done, None


def pos_to_representation_smaller_board(position): 
    "preprocesses chess position for CNN with one-hot-encoding"
    pieces_pos = position.piece_map()
    pieces_pos = {v: k for k, v in pieces_pos.items()}

    if position.is_game_over():
        R = False
        K, k = pieces_pos[chess.Piece.from_symbol('K')], pieces_pos[chess.Piece.from_symbol('k')]
    else:
        K, R, k = pieces_pos[chess.Piece.from_symbol('K')], pieces_pos[chess.Piece.from_symbol('R')], pieces_pos[chess.Piece.from_symbol('k')]

    position_repr = np.zeros((4,4))
    # white_king = np.zeros((4,4))
    # white_rook = np.zeros((4,4))
    # black_king = np.zeros((4,4))

    position_repr[chess.square_rank(K)][chess.square_file(K)] = 1
    position_repr[chess.square_rank(k)][chess.square_file(k)] = -1

    # white_king[chess.square_rank(K)][chess.square_file(K)] = 1
    # black_king[chess.square_rank(k)][chess.square_file(k)] = 1

    if R:
        position_repr[chess.square_rank(R)][chess.square_file(R)] = 0.5
        # white_rook[chess.square_rank(R)][chess.square_file(R)] = 1

    return np.array(position_repr).flatten() # return np.array([white_king, white_rook, black_king]).reshape((3,4,4)).flatten() # dit was eers 8,8,3!


def rep_to_move_smaller_board(repr:int, current_position, env):
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

        squares = [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]

        if new_square in squares: # check if new_square is on the board
            move = chess.Move(king_position, new_square)
            if move in legal_moves_smaller_board(squares, env):
                return move, True



    elif 7 < repr <= 19:
        piece = chess.Piece.from_symbol('R')
        rook_position = pieces_pos[piece]
        rank, file = chess.square_rank(rook_position),chess.square_file(rook_position)

        if repr <= 10:
            d_file, d_rank = 0, repr-7
        elif 11 <= repr <= 13:
            d_file, d_rank = repr-10, 0
        elif 14 <= repr <= 16:
            d_file, d_rank = 0, 13-repr
        elif 17 <= repr <= 19:
            d_file, d_rank = 16-repr, 0

        new_rank, new_file = rank+d_rank, file+d_file
        new_square = chess.square(new_file, new_rank)
        
        if new_square in [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]: # check if new_square is on the smaller board
            move = chess.Move(rook_position, new_square)
            if move in env.legal_moves: # so if move within the board and legal
                return move, True
    
    return random.choice(legal_moves_smaller_board(squares, env)), False # the move is not legal, so return random move and False


def move_to_rep_smaller_board(move, env):
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
                return d_rank + 7
            else:
                return -d_rank + 13
        else:
            if d_file > 0:
                return d_file + 10
            else:
                return -d_file + 16