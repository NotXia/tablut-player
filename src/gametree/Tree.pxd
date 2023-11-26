from .State cimport State
from .TreeNode cimport TreeNode
from .utils cimport *
from libc.time cimport time_t

cdef class Tree():
    cdef State state
    cdef char player_color
    cdef TreeNode root
    cdef int turns_count

    cdef float[:] early_positive_weights
    cdef float[:] early_negative_weights
    cdef float[:] mid_positive_weights
    cdef float[:] mid_negative_weights
    cdef float[:] late_positive_weights
    cdef float[:] late_negative_weights
    cdef float[:] curr_positive_weights
    cdef float[:] curr_negative_weights

    cdef int __explored_nodes

    cdef void __updateWeights(self)
    cpdef tuple[Coord, Coord, score_t] decide(self, int timeout)
    cdef score_t minimax(self, TreeNode tree_node, int max_depth, score_t alpha, score_t beta, time_t timeout_timestamp)
