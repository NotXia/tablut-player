"""
    Class that represents a node in the game tree.
"""
cdef class TreeNode:
    def __init__(self, Coord start, Coord end):
        self.start = start
        self.end = end
        self.score = 0
        self.children = []
        self.critical_len = 0


    """
        Returns a generator of the children of the node with a given state.
        Children are generated if needed.

        Parameters
        ----------
            state : State
                State of the board. Needed to generate the children if needed.

        Returns
        -------
            children_generator : Generator[TreeNode]
                Children of this node.
    """
    cdef list[TreeNode] getChildren(self, State state):
        cdef list[Move] critical_moves, other_moves
        cdef Coord start, end
        cdef TreeNode child

        if len(self.children) == 0:
            # Children of this node haven't been generated yet
            critical_moves, other_moves = state.getMoves()
            self.critical_len = len(critical_moves)
            for start, end in critical_moves + other_moves:
                child = TreeNode(start, end)
                self.children.append(child)
                
        # Children of this node already generated
        return self.children