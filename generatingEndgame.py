import chess
import random

def generate_random_position(squares, pieces):
    board = chess.Board()
    board.clear()

    # Position King randomly, then rook and other king
    king_square = random.choice(squares) #random.choice(chess.SQUARES)
    board.set_piece_at(king_square, chess.Piece(chess.KING, chess.WHITE))

    enemy_king_square = random.choice([sq for sq in squares if sq != king_square])
    board.set_piece_at(enemy_king_square, chess.Piece(chess.KING, chess.BLACK))

    piece_squares = [king_square, enemy_king_square]
    for piece in pieces:
        piece_square = random.choice([sq for sq in squares if sq not in piece_squares])
        board.set_piece_at(piece_square, chess.Piece(piece, chess.WHITE))
        piece_squares.append(piece_square)

    return board

def is_legal_position(board):
    return board.is_valid()

def play_random_game(board):
    while not board.is_game_over():
        legal_moves = list(board.legal_moves)
        random_move = random.choice(legal_moves)
        board.push(random_move)

    return board.result()


def generate_endgame_FEN(squares, pieces):

    legal_position = False
    game_over = False
    while not legal_position or game_over:
        starting_position = generate_random_position(squares, pieces)
        legal_position = is_legal_position(starting_position)
        game_over = starting_position.is_game_over()

    return starting_position.fen()

