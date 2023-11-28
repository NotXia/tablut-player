# distutils: language = c++
from .State cimport State
from .utils cimport *
from libcpp.unordered_map cimport unordered_map
from libcpp.queue cimport queue


cdef char EXACT = 0
cdef char LOWERBOUND = 1
cdef char UPPERBOUND = 2

cdef TraspositionEntry INV_ENTRY
INV_ENTRY.depth = -1
INV_ENTRY.entry_type = 0
INV_ENTRY.value = 0


cdef class TranspositionTable:
    """
        Parameters
        ----------
            max_size : int
                The maximum number of entries the table can hold.
    """
    def __init__(self, unsigned int max_size):
        self.max_size = max_size
        self.curr_size = 0


    cdef void setEntry(self, State state, char entry_type, score_t value, int depth):
        cdef int board_hash = state.hash()
        cdef TraspositionEntry entry
        entry.entry_type = entry_type
        entry.value = value
        entry.depth = depth

        if (self.table.find(board_hash) == self.table.end()):
            if self.curr_size+1 > self.max_size:
                self.table.erase(self.drop_queue.front())
                self.drop_queue.pop()
                self.curr_size -= 1
            self.drop_queue.push(board_hash)
            self.curr_size += 1
        self.table[board_hash] = entry
        

    cdef TraspositionEntry getEntry(self, State state):
        cdef int board_hash = state.hash()
        if (self.table.find(board_hash) == self.table.end()):
            return INV_ENTRY
        return self.table[board_hash]
    