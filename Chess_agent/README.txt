# How to train and play

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
