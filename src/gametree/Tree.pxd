from .State cimport State
from .TreeNode cimport TreeNode
from .utils cimport *

cdef class Tree():
    cdef State state
    cdef char player_color
    cdef TreeNode root
    cdef int __explored_nodes

    cdef score_t minimax(self, TreeNode tree_node, int max_depth, score_t alpha, score_t beta, int timeout_timestamp)
