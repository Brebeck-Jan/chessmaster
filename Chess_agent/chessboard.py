import chess
import chess.svg
import numpy as np

from IPython.display import SVG, display

# set up mapper for chessman for specific layer
mapper = {}

# Pawn (Bauer)
mapper["p"] = 0
mapper["P"] = 0

# Rook (Turm)
mapper["r"] = 1
mapper["R"] = 1

# Knight (Springer)
mapper["n"] = 2
mapper["N"] = 2

# Bishop (Läufer)
mapper["b"] = 3
mapper["B"] = 3

# Queen (Königin)
mapper["q"] = 4
mapper["Q"] = 4

# King (König)
mapper["k"] = 5
mapper["K"] = 5

class Board(object):

    def __init__(self):
        """
        Setup Chess Environment
        """

        self.board = chess.Board()

        self.init_layer_board()
    
    def get_legal_move(self, all = False):
        """
        Return random legal move.
        """

        # get all legal moves
        legal_moves = [move for move in self.board.generate_legal_moves()]

        if all:

            # return all possible legal moves
            return legal_moves

        # get random move
        if legal_moves:

            ran_move = np.random.choice(legal_moves)

        else:

            # Check Mate!
            ran_move = None

        # return random move
        return ran_move

    def init_layer_board(self):
        """
        Initialize a numerical representation of the Board.
        Each chessman class should be represented at an own layer.
        """

        self.layer_board = np.zeros(shape = (8, 8, 8))

        # loop through board
        for i in range(64):

            row = i // 8
            col = i % 8

            piece = self.board.piece_at(i)

            # No piece at given position
            if piece == None:
                continue

            # check colour
            elif piece.symbol().isupper():

                # is white
                colour = 1
            
            else:

                # is black
                colour = -1

            # map piece
            layer = mapper[piece.symbol()]

            # fill layer board
            self.layer_board[layer, row, col] = colour
            self.layer_board[6, :, :] = 1 / self.board.fullmove_number
        
        # mark Layer 6, row 0 with colour | UNDERSTAND
        if self.board.turn:

            self.layer_board[6, 0, :] = 1
        else:
            
            self.layer_board[6, 0, :] = -1

        # update layer 7 | UNDERSTAND
        self.layer_board[7, :, :] = 1 
    
    def update_layer_board(self):
        """
        Save old layer board and create new.
        """

        self._prev_layer_board = self.layer_board.copy()
        self.init_layer_board()
    
    def get_prev_layer_board(self):
        """
        Return to old layer board.
        """

        self.layer_board = self._prev_layer_board.copy()
        self._prev_layer_board = None
    
    def move(self, move, capture_reward_factor = 0.01):
        """
        Run a chess move.
        Return if game has ended and the material balance delta.
        capture_reward_factor adjusts reward for capturing figures.
        """

        # get actual material balance
        act_balance = self.get_material_value()

        # run move
        self.board.push(move)

        # update layer_board
        self.update_layer_board()

        # get new material balance
        new_balance = self.get_material_value()

        # calculate material balance delta
        balance_reward = (new_balance - act_balance) * capture_reward_factor

        # retrieve result from board
        result = self.board.result()

        # check if game is over and set corresponding reward
        if result == "*":
            
            # game is not over
            reward = 0
            end = False
        
        elif result == "1-0":

            # white (AI only plays white) wins
            reward = 100 # check whcih number is the best (original is 1, -1)
            end = True
        
        elif result == "0-1":

            # Black (opponent) wins
            reward = -100 # siehe oben
            end = True
        
        elif result == "1/2-1/2":

            # Draw
            reward = 0 # OPTION: set to -0.5 to train programm to win!?
            end = True

        # add reward and balance reward
        reward += balance_reward

        # return
        return end, reward
    
    def handle_user_move(self, user_input):
        """
        Transform user input in python-chess move.
        """

        # transform to python-chess move
        move = chess.Move.from_uci(user_input)

        # check if move is legal
        moves = self.get_legal_move(all=True)
        
        if move in moves:

            # move is legal
            return move
        
        else:

            # move is illegal
            raise ValueError("move is illegal")

    def get_material_value(self, option = "all"):
        """
        Get the material values.
        Optional: only return material value for spec figure.
        Values are based on changevalue to pawns (Wikipedia).
        """

        # get values
        pawn_value  = 1 * np.sum(self.layer_board[0, :, :]) # Pawns have value 1
        minor_value = 3 * np.sum(self.layer_board[2:4, :, :]) # Minors have value 3
        rook_value  = 5 * np.sum(self.layer_board[1, :, :]) # Rooks have value 5
        queen_value = 9 * np.sum(self.layer_board[4, :, :]) # Queen has value 9

        # return values
        if option == "all":

            # return all
            return pawn_value + minor_value + rook_value + queen_value
        
        elif option == "pawn":

            return pawn_value
        
        elif option == "minor":

            return minor_value
        
        elif option == "rook":

            return rook_value
        
        elif option == "queen":

            return queen_value
        
        else:

            # No valid option selected
            return 0

    def reset_board(self):
        """
        Reset the chess board
        """

        self.board = chess.Board()
        self.init_layer_board()
    
    def show_board(self):
        """
        Show svg version of board
        """

        display(SVG(chess.svg.board(board = self.board)))
