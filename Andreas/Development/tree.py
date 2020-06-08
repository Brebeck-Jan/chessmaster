import numpy as np

# setup Node Structure for MCTS

class Node(object):

    def __init__(self, parent = None):
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
    
    def update(self, value):
        """
        Update recursivley value of Node and parent node
        """

        # update own value
        self.value += value

        # update parent value
        if self.parent:
            self.parent.update(value)
    
    def select(self):
        """
        Recursivley choose node with highest value.
        Use Upper-Confidence-Bound Algorithm to reduce Exploration / Exploitation Trade off.
        """
        
        def UCB1(self, c = 2):
            """
            UCB1-Algorithm
            TASK: Check how to choose optimal c
            """

            # mean value
            v = self.value / self.visited

            # calculate UCB
            ucb = v + c * ( np.sqrt( np.ln( self.parent.visited ) / self.visited ) )

            # return UCB
            return ucb
        
        # Setup dict for child selection
        child_ucb = {}

        for child in self.children:
            
            child_ucb[child] = UCB1(child)
        
        # select highest value
        child = max(child_ucb, key = child_ucb.get)

        # rerun search if node has childs
        if child.children:
            
            child.children.select(self)
        
        # no child so return node
        else:
            return child
    
    def expanse(self):
        """
        Expand nodes.
        Add one child node for every possible action from this point.
        """
        pass

    def rollout(self):
        """
        Simulate one complete Playout for node and get value!
        """
        pass

