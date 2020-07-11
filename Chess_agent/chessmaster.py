from game import Chess
from agent import Agent
import pandas as pd
import matplotlib.pyplot as plt

class Chessmaster():

    def __init__(self):

        self.agent = Agent()

        self.chess = Chess(self.agent)
    
    def train_agent(self, iterations, maxmoves = 60, update_rate = 5):
        """
        Call train function in game.py
        """

        # load old model
        self.load_model()

        #train
        self.chess.start_learning(iters = iterations, update_rate = update_rate, max_moves= maxmoves)

        #save model
        self.save_model()


    def play(self):
        """
        Play against chessmaster
        """

        # load model
        self.load_model()

        # play
        self.chess.play_against_chessmaster()

    def save_model(self):
        """
        Save model to file
        """

        self.agent.save_model()
    
    def load_model(self):
        """
        Load model from file
        """

        # load old model if exists
        try:

            self.agent.load_model()
            print("Loaded model")

        except:

            # No model exists
            print("No model found")
            pass
        
    
    def plot_reward_trace(self):
        """
        Plot reward trace of model
        """

        reward = pd.DataFrame(self.chess.reward_trace)
        reward.rolling(window = 500, min_periods = 0).mean().plot(figsize = (16, 9), title = "Average Reward")

        # show
        plt.show()

    def plot_balance_trace(self):
        """
        Plot Balance trace of model
        """

        reward = pd.DataFrame(self.chess.balance_trace)
        reward.rolling(window = 500, min_periods = 0).mean().plot(figsize = (16, 9), title = "Average Balance")

        # show
        plt.show()
        