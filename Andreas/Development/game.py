from chessboard import Board
from tree import Node
import re

class Chess(object):

    def __init__(self, playercolour = "Black"):

        self.board = Board()
        self.playercolour = playercolour
    
    def play(self):
        """
        Start a chessgame.
        """

        # setup player who starts
        turn = "Player" if self.playercolour == "White" else "AI"
        print("Starting turn: ", turn)

        end = False
        number = 0

        # print initial field
        print("Welcome to your chess game:\n", self.board.board)

        while not end:

            number += 1

            # AIs Turn
            if turn == "AI":
                
                # get AI move
                move = self.agent_step()

                # pass move to Player
                turn = "Player"
            
            elif turn == "Player":
                
                # get player move
                move = self.player_step()

                # pass move to AI
                turn = "AI"

            # execute move and check if game ended with move
            end, reward = self.board.move(move)

            print(50*"-")
            print("Actual Board in Turn: ", number)
            print(self.board.board)
    
    def player_step(self):
        """
        Let player make move.
        """
        
        print("Enter fromsquare:tosquare, e.g. (a2a3)")

        # get input
        user_input = input("Type your move here: ")

        # check if input is valid
        pattern = re.compile("[a-h][1-8][a-h][1-8]")

        if bool(pattern.match(user_input)):

            # move is legal | transform to python-chess move
            move = self.board.handle_user_move(user_input)

        else:

            raise ValueError(f"Input {user_input} wasn't in right format.")
            
        # return move
        return move

        
    
    def agent_step(self, depth = 120):
        """
        Let agent (now MCTS) make a step => update to Neural Network!
        """
        
        # get root node of actual game
        self.root = Node(board = self.board)

        # initial expand
        self.root.expand()

        # initial rollout
        a = list(self.root.children.keys())[0]
        self.root.children[a].rollout(depth=depth)

        # get move from Monte Carlo Tree Search
        move = self.mcts(100)

        # return move
        return move
    
    def reset(self):
        """
        Reset Chessboard to start new Game.
        """

        # delete chessboard
        self.board.reset_board()
    
    def mcts(self, iterations, printable = False):
        """
        Monte Carlo Tree Search
        """

        for i in range(iterations):

            # loop through MCTS
            leaf = self.root.select(printable = printable)
        
        # return childs (moves) with values
        result = {}
        for child in self.root.children:

            result[self.root.children[child].move] = self.root.children[child].value
        
        # get max move
        move = max(result, key = result.get)

        # return move
        return move