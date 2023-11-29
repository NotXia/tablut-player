ctypedef int pos_t
ctypedef tuple[pos_t, pos_t] Coord
ctypedef tuple[Coord, Coord] Move

ctypedef float score_t


cdef Coord NULL_COORD


cdef double getTime()