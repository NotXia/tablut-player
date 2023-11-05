from __future__ import annotations

"""
    Class that represents a node in the game tree.
"""
class TreeNode():
    def __init__(self, start:tuple[int, int], end:tuple[int, int], parent: TreeNode):
        self.start = start
        self.end = end
        self.score: float = None
        self.parent = parent
        self.children: list[TreeNode] = []