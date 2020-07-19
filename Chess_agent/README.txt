# How to use chessmaster.py (easy) | In Jupyter Notebook!

from chessmaster import Chessmaster

# set instance
chessmaster = Chessmasetr()

# play
chessmaster.play()

----------------------------------------------

# How to use game.py

from agent import Agent
from game import Chess

agent = Agent()
chess = Chess(agent = agent)

iterations = 20

# train model
chess.start_learning(iters = iterations)

# save model
chess.agent.save_model()

# play against model
chess.play_against_chessmaster()
