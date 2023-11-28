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
    cdef char[:, :] memv_board
    cdef bint is_white_turn
    cdef unsigned short N_ROWS
    cdef unsigned short N_COLS
    cdef unsigned short N_WHITES
    cdef unsigned short N_BLACKS 
    cdef int MAX_DIST_TO_KING
    cdef int MAX_DIST_TO_ESCAPE

    cdef int hash(self, bint normalize=*)
    cdef cnp.ndarray getNormalizedBoard(self)

    cdef Coord __findKing(self)
    cdef char getGameState(self)
    cdef bint isValidCell(self, pos_t i, pos_t j)
    cdef bint isWall(self, pos_t i, pos_t j)
    cdef bint isObstacle(self, pos_t i, pos_t j, char num_camp=*)
    cdef char getCampOfPawnAt(self, pos_t i, pos_t j)

    cdef tuple[list[Move], list[Move]] getMoves(self)
    cdef list[Move] __getPawnMoves(self, pos_t i, pos_t j)
    cdef int numSteps(self, pos_t i, pos_t j, char direction)

    cdef bint isCaptured(self, pos_t i, pos_t j, char to_filter_axis=*)
    cdef bint isCapturingElementFor(self, pos_t pawn_i, pos_t pawn_j, pos_t check_i, pos_t check_j)

    cdef list[tuple[Coord, char]] applyMove(self, Coord start, Coord end)
    cdef void revertMove(self, Coord old_start, Coord old_end, list[tuple[Coord, char]] captured)

    cdef score_t evaluate(self, char player_color, int max_depth, float[:] positive_weights, float[:] negative_weights)
    cdef score_t heuristics(self, char player_color, float[:] positive_weights, float[:] negative_weights)
    cdef score_t __pawnRatio(self, char color)
    cdef score_t __avgProximityToKingRatio(self, char color)
    cdef score_t __safenessRatio(self, char color)
    cdef score_t __minDistanceToEscapeRatio(self)

