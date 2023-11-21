from .utils cimport *
from .State cimport State

cdef class TreeNode:
    cdef Coord start
    cdef Coord end
    cdef score_t score
    cdef list[TreeNode] children
    cdef unsigned int critical_len

    cdef list[TreeNode] getChildren(self, State state)