# distutils: language = c++
from .State cimport State
from .utils cimport *
from libcpp.unordered_map cimport unordered_map
from libcpp.queue cimport queue


cdef char EXACT
cdef char LOWERBOUND
cdef char UPPERBOUND


cdef struct TraspositionEntry:
    char entry_type
    score_t value
    int depth


cdef class TranspositionTable:
    cdef unsigned int max_size
    cdef unsigned int curr_size
    cdef unordered_map[int, TraspositionEntry] table
    cdef queue[int] drop_queue

    cdef void setEntry(self, State state, char entry_type, score_t value, int depth)
    cdef TraspositionEntry getEntry(self, State state)
