import chessboard
import tree

board = chessboard.Board("ads")
root = tree.Node(board = board)
root.expand()
a = list(root.children.keys())[0]
root.children[a].rollout(depth=120)

def mcts(iterations):
    
    for i in range(iterations):

        print("iter: ", i)

        # select leaf
        leaf = root.select()

        print(50*"-")
        # [print(c.value) for c in root.children.values()]
    
    # return childs (moves) with values
    result = {}
    for child in root.children:

        result[root.children[child].move] = root.children[child].value
    
    print(result)
    print("maxmove", max(result, key = result.get))
    # return result