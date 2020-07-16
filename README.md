# Chessmaster
by  
- Thorsten Hilbradt,
- Andreas Bernrieder 7876007,
- Niklas Wichter, 
- Jan Brebeck

### How to use
Navigate into /Chess_agent

in console run: 
  pip install -r requirements.txt
to install all requiered packages.

Run code:
  - in jupyter notebook
    Open Graphical.ipybn in jupyter notebook / jupyter lab and run required cells
  - in python
    run python in console
    
    python code:
    
    from chessmaster import Chessmaster
    
    chessmaster = Chessmaster()
    
    // function to train chessmaster\br
    chessmaster.train_agent(iterations = 100)
    
    // function to play against chesssmaster\br
    chessmaster.play()
    
    // alternative
    from game import Chess
    from agent import Agent
    
    // define rl agent
    agent = Agent()
    
    // define chess
    chess = Chess(agent)
    
    // start training
    chess.start_learning(iters = 100)
    
    // play against pure MCST
    chess.play_against_mcts
    
    // play against trained model
    chess.play_against_chessmaster()
