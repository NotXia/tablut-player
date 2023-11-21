import numpy as np
cimport numpy as cnp
from .utils cimport *

cdef score_t MAX_SCORE
cdef score_t MIN_SCORE

cdef char EMPTY
cdef char BLACK
cdef char WHITE
cdef char KING 

cdef char WHITE_WIN
cdef char BLACK_WIN
cdef char OPEN

cdef char UP
cdef char DOWN
cdef char RIGHT
cdef char LEFT

cdef char VERTICAL
cdef char HORIZONTAL
cdef char VERT_HORIZ


cdef class State:
    cdef cnp.ndarray board
    cdef bint is_white_turn
    cdef unsigned short N_ROWS
    cdef unsigned short N_COLS
    cdef unsigned short N_WHITES
    cdef unsigned short N_BLACKS 

    cdef char getGameState(self)
    cdef bint isValidCell(self, int i, int j)
    cdef bint isWall(self, int i, int j)
    cdef bint isObstacle(self, int i, int j, char num_camp=*)
    cdef char getCampOfPawnAt(self, int i, int j)

    cdef tuple[list[Move], list[Move]] getMoves(self)
    cdef list[Move] __getPawnMoves(self, int i, int j)
    cdef int numSteps(self, int i, int j, char direction)

    cdef bint isCaptured(self, int i, int j, char to_filter_axis=*)
    cdef bint isCapturingElementFor(self, int pawn_i, int pawn_j, int check_i, int check_j)

    cdef list[tuple[Coord, char]] applyMove(self, Coord start, Coord end)
    cdef void revertMove(self, Coord old_start, Coord old_end, list[tuple[Coord, char]] captured)

    cdef score_t evaluate(self, char player_color)
    cdef score_t heuristics(self, char player_color)
    cdef int __countPawn(self, char color)
    cdef score_t __threatRatio(self, char color)