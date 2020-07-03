from game import Chess
from agent import Agent

class Chessmaster():

    def __init__(self):

        self.agent = Agent()

        self.chess = Chess(self.agent)
    
    def train_agent(self, iterations, maxmoves = 60, update_rate = 5):
        """
        Call train function in game.py
        """

        self.chess.start_learning(iters = iterations, update_rate = update_rate, max_moves= maxmoves)

    def show_learn_trace(self):
        """
        Plot curves about learn rate, etc.
        """

        # implementation missing!
        pass

    def save_model(self):
        """
        Save model to file
        """

        self.agent.save_model()
    
    def load_model(self):
        """
        Load model from file
        """

        self.agent.load_model()