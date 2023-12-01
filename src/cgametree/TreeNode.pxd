from .utils cimport *
from .State cimport State

cdef class TreeNode:
    cdef Coord start
    cdef Coord end
    cdef score_t score
    cdef list[TreeNode] children
    cdef unsigned int critical_len

    cdef void generateChildren(self, State state, double timeout_timestamp)
    cdef prioritizeChild(self, int index)