from chessboard import Board
from tree import Node
import re
import gc

class Chess(object):

    def __init__(self, playercolour = "Black", agent):

        self.board = Board()
        self.playercolour = playercolour

        # Get Chess agent
        self.chess_agent = agent

        # set up parameters
        self.gamma = 0.9 # E [0,1]
        self.memsize = 2000
        self.batch_size = 256

        # set up arrays for neural network learning
        self.mem_state = np.zeros(shape = (1, 8, 8, 8))
        self.mem_sucstate = np.zeros(shape = (1, 8, 8, 8))
        self.mem_reward = np.zeros(shape = (1))
        self.mem_error = np.zeros(shape = (1))
        self.mem_episode_active = np.ones(shape = (1))
    
    def play_against_mcts(self):
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
                move = self.mcts_step()

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
    
    def train(self):
        """ 
        Method to train the Chess agent.
        """

        # keep track of turncounts
        turncount = 0

        end = False

        while not end:

            # get state and predicted state value
            state = np.expand_dims(self.board.layer_board.copy(), axis = 0)
            state_value = self.chess_agent.predict(state)
            
            # agent plays as White player
            if self.board.board.turn:
                move = self.mcts_step()
            
            # use myopic agent as Black player
            else:
                move = myopic_agent_step()

            # make move
            end, reward = self.board.move(move)

            # get sucstate and predict sucstate value
            sucstate = np.expand_dims(self.board.layer_board.copy(), axis = 0)
            suc_state_value = self.chess_agent.predict(sucstate)

            # calculate error
            error = reward + self.gamme * suc_state_value - state_value
            error = np.float(np.squeeze(error))

            # add 1 to turncount
            turncount += 1

            # save if episode is active
            episode_active = 0 if end else 1

            # construct training sample
            self.mem_state = np.append(self.mem_state, state, axis = 0)
            self.mem_reward = np.append(self.mem_reward, reward)
            self.mem_sucstate = np.append(self.mem_sucstate, sucstate, axis = 0)
            self.mem_error = np.append(self.mem_error, error)
            self.mem_episode_active = np.append(self.mem_episode_active, episode_active)

            # clear memory if neccessary?
            if self.mem_state.shape[0] > self.memsize:
                
                self.mem_state = self.mem_state[1:]
                self.mem_reward = self.mem_reward[1:]
                self.mem_sucstate = self.mem_sucstate[1:]
                self.mem_error = self.mem_error[1:]
                self.mem_episode_active = self.mem_episode_active[1:]
            
            # update agent every 10 steps
            if turncount % 10 == 0:

                self.update_agent()

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

    def mcts_step(self, depth = 120):
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
    
    def myopic_agent_step(self):
        """
        Computer Agent who plays myopic.
        """

        move_dict = {}

        for move in self.board.get_legal_move(all = True):

            # get old material value
            material_old = self.board.get_material_value()
            
            # run move
            self.board.board.push(move)

            # get new material value
            material_new = self.board.get_material_value()

            # undo move
            self.board.board.pop()

            # save move with value
            move_dict[move] = material_new - material_old
        
        # get max move
        max_move = max(move_dict, key = move_dict.get)

        # return move
        return max_move
    
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
    
    def update_agent(self):
        """
        Update Chess agent.
        """

        # get data
        choice_indices, states, rewards, sucstates, episode_active = self.get_minibatch()

        # calculate TD error
        td_errors = self.agent.TD_update(states, rewards, sucstates, episode_active, gamma = self.gamma)

        # update mem error
        self.mem_error[choice_indices.tolist()] = td_errors
    
    def get_minibatch(self):
        """
        Get minibatch of experienced games.
        """

        # set sampling priorities
        sampling_priorities = np.abs(self.mem_error) + 1e-9

        sampling_probs = sampling_priorities / np.sum(sampling_priorities)
        sample_indices = [x for x in range(self.mem_state.shape[0])]

        # set choice indices
        choice_indices = np.random.choice(
            sample_indices,
            min(self.mem_state.shape[0], self.batch_size),
            p = np.squeeze(sampling_probs),
            replace = False
        )

        # get data
        states = self.mem_state[choice_indices]
        rewards = self.mem_reward[choice_indices]
        sucstates = self.mem_sucstate[choice_indices]
        episode_active = self.mem_episode_active[choice_indices]

        # return
        return choice_indices, states, rewards, sucstates, episode_active

        """
        Funktionsweise:
        neuronales netz lernt die state vlaues vorherzusagen, 
        damit kann der mcts um einiges schneller errechnet werden!!!!! 
        einfach nach der expansion die zugehörigen state values von dem neuronalen 
        netz geben lassen!!

        Was fehlt: der anfang von der learn. py also die iterationen und dann das updatedn des fiixed models
        dann noch dass der mcts nicht komplett resettet wird jedesmal, sondern, dass die gewählte child node neue parent jnode wird und die werte erhalten bleiben!
        """