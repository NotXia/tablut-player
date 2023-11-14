from __future__ import annotations
from .State import State
from typing import Generator
import cython
import logging
logger = logging.getLogger(__name__)
if not cython.compiled: logger.warn(f"Using non-compiled {__file__} module")

"""
    Class that represents a node in the game tree.
"""
class TreeNode():
    def __init__(self, start:tuple[int, int]|None, end:tuple[int, int]|None, parent: TreeNode|None):
        self.start = start
        self.end = end
        self.score: float = None
        self.parent = parent
        self.children: list[TreeNode] = None


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
    def getChildren(self, state: State) -> Generator[TreeNode]:
        if self.children is None:
            # Children of this node haven't been generated yet
            self.children = []
            
            for start, end in state.getMoves():
                child = TreeNode(start, end, self)
                self.children.append(child)
                
        # Children of this node already generated
        for child in self.children:
            yield child