from .utils cimport getTime

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


    cdef void generateChildren(self, State state, double timeout_timestamp):
        cdef list[Move] critical_moves, other_moves
        cdef Coord start, end
        cdef TreeNode child

        if len(self.children) == 0:
            critical_moves, other_moves = state.getMoves()
            self.critical_len = len(critical_moves)
            for start, end in critical_moves + other_moves:
                if getTime() >= timeout_timestamp:
                    self.children = []
                    self.critical_len = 0
                    return
                child = TreeNode(start, end)
                self.children.append(child)

    cdef prioritizeChild(self, int index):
        self.children.insert(0, self.children.pop(index))