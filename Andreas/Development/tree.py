import numpy as np
import copy

# setup Node Structure for MCTS

class Node(object):

    def __init__(self, parent = None, board = None, move = None):
        """
        Node for MCTS. 
        Saves: 
        - value
        - parent Node
        - Children Nodes
        - times visited
        """

        # child nodes
        self.children = {}

        # parent node
        self.parent = parent

        # value
        self.value = 0

        # visited
        self.visited = 0

        # chess board
        self.board = board

        # assigned move
        self.move = move
    
    def set_value(self, value):
        """
        Set value of node. Used by agent
        """
        self.value = value
    
    def update(self, value):
        """
        Update recursivley value of Node and parent node
        """

        # update own value
        self.value += value

        # node was visited
        self.visited += 1

        # update parent value
        if self.parent:
            self.parent.update(value)
    
    def select(self, printable=False):
        """
        Recursivley choose node with highest value.
        Use Upper-Confidence-Bound Algorithm to reduce Exploration / Exploitation Trade off.
        """
        
        def UCB1(self, c = 2):
            """
            UCB1-Algorithm
            TASK: Check how to choose optimal c
            """

            # calculate UCB
            try:

                # mean value
                v = self.value / self.visited

                ucb = v + c * ( np.sqrt( np.log( self.parent.visited ) / self.visited ) )
            
            except ZeroDivisionError:

                # if node wasn't visited before set high value => Explorationa
                ucb = 1000000

            # return UCB
            return ucb
        
        # Setup dict for child selection
        child_ucb = {}

        for child in list(self.children.values()):
            
            child_ucb[child] = UCB1(child)
        
        # select highest value
        child = max(child_ucb, key = child_ucb.get)
        print("max child", child) if printable else 0

        # rerun search if node has childs
        if child.children:
            
            child.select(printable = printable)
        
        # no child so return node
        else:

            # integration of full functional MCTS

            # prints
            print("node move:", child.move) if printable else None
            print("board:\n", child.board.board) if printable else None

            # 2. Step: Expansion
            child.expand()

            # 3. Step: Rollout of one node
            a = list(child.children.keys())[0]
            child.children[a].rollout(depth=60, printable = printable) # make depth more abstract and check hich depth is nice!

            return child
    
    def expand(self):
        """
        Expand nodes.
        Add one child node for every possible action from this point.
        """
        
        # get list of all possible moves
        moves = self.board.get_legal_move(all = True)

        for move in moves:

            child = Node(parent = self, board = copy.deepcopy(self.board), move = move)
            child.board.move(child.move)
            self.children[move] = child

    def rollout(self, depth = None, printable = False):
        """
        Simulate one complete Playout for node and get value!
        Set up depth means limit maximum depth of simulation
        """
        
        # simulate with random moves for both players

        # save old board
        _prev_board = copy.deepcopy(self.board)

        value = 0
        if depth:

            end = False
            while not end:

                move = self.board.get_legal_move()
                end, reward = self.board.move(move)
                value += reward
        
        else:

            for i in range(depth):

                move = self.board.get_legal_move()
                end, reward = self.board.move(move)
                value += reward
        
        self.update(value)

        # reset to old board
        print("after rollout:\n", self.board.board) if printable else None
        self.board = copy.deepcopy(_prev_board)
        _prev_board = None