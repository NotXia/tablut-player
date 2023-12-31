from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Generator
import random
import math
import cython
import logging
logger = logging.getLogger(__name__)
if not cython.compiled: logger.warning(f"Using non-compiled {__file__} module")

random.seed(42)
np.random.seed(42)


MAX_SCORE = 1000
MIN_SCORE = -1000

EMPTY = 0
BLACK = 1
WHITE = 2
KING  = 3

WHITE_WIN = 4
BLACK_WIN = 5
OPEN =  6

UP = 7
DOWN = 8
RIGHT = 9
LEFT = 10

VERTICAL = 11
HORIZONTAL = 12

# Define the winning tiles
ESCAPE_TILES = [
    (0, 1), (0, 2), (0, 6), (0, 7),
    (1, 0), (2, 0), (6, 0), (7, 0),
    (1, 8), (2, 8), (6, 8), (7, 8),
    (8, 1), (8, 2), (8, 6), (8, 7)
]
# Associate each camp with a number
CAMP_DICT = {
    (0, 3): 1,
    (0, 4): 1,
    (0, 5): 1,
    (1, 4): 1,
    (3, 0): 2,
    (4, 0): 2,
    (5, 0): 2,
    (4, 1): 2,
    (3, 8): 3,
    (4, 8): 3,
    (5, 8): 3,
    (4, 7): 3,
    (8, 3): 0,
    (8, 4): 0,
    (8, 5): 0,
    (7, 4): 0
}
CASTLE_TILE = (4, 4)
NEAR_CASTLE_TILES = [(3, 4), (5, 4), (4, 3), (4, 5)]

zobrist_table = np.random.randint(-1e8, 1e8, size=(9, 9, 3))
zobrist_black = random.randint(-1e8, 1e8)


def zobristHash(board, n_rows, n_cols, is_white_turn):
    state_hash = 0

    if not is_white_turn: state_hash ^= zobrist_black
    for i in range(n_rows):
        for j in range(n_cols):
            if board[i, j] == KING: state_hash ^= zobrist_table[i, j, 0]
            elif board[i, j] == WHITE: state_hash ^= zobrist_table[i, j, 1]
            elif board[i, j] == BLACK: state_hash ^= zobrist_table[i, j, 2]

    return int(state_hash)


"""
    Simple class to represent the state of the game.
    The state is represented by a matrix of bytes of dimensions 9x9.
"""
class State():
    def __init__(self, board: npt.NDArray[np.byte], is_white_turn: bool, rules="ashton"):
        self.board = board
        self.is_white_turn = is_white_turn

        if rules == "ashton":
            self.N_ROWS = 9
            self.N_COLS = 9
            self.N_WHITES = 8
            self.N_BLACKS = 16
            self.MAX_DIST_TO_KING = 14 # Maximum distance between a pawn and the king
            self.MAX_DIST_TO_ESCAPE = 13 # Maximum distance between the king and the farthest escape tile
        else:
            raise ValueError("Unknown rules")


    def __str__(self):
        return f"WhiteTurn = {self.is_white_turn}\n {str(self.board)}"


    def __hash__(self):
        return self.hash(normalize=True)


    """
        Computes the Zobrist hash of the current state.

        Parameters
        ----------
            normalize : bool
                If True, the board will be normalized with rotations and flips.
                If False, the hash will be computed on the board as is.

        Returns
        -------
            hash : int
    """
    def hash(self, normalize=False):
        if normalize:
            return zobristHash(self.getNormalizedBoard(), self.N_ROWS, self.N_COLS, self.is_white_turn)
        else:
            return zobristHash(self.board, self.N_ROWS, self.N_COLS, self.is_white_turn)

    
    """
        Produces a normalized version of the current board
        where the king is in the first upper-quadrant.

        Returns
        -------
            normalized_board : np.array
    """
    def getNormalizedBoard(self):
        pos_king = np.argwhere(self.board == KING)
        if len(pos_king) == 0: return self.board
        pos_king = tuple(pos_king[0])

        if (pos_king[0] <= math.floor(self.N_ROWS/2)) and (pos_king[1] >= math.ceil(self.N_COLS/2)):
            if pos_king[0] + pos_king[1] >= self.N_COLS-1:
                return np.rot90(self.board, k=1)
            else:
                return np.flip(self.board, axis=1)
        elif (pos_king[0] >= math.ceil(self.N_ROWS/2)) and (pos_king[1] >= math.floor(self.N_COLS/2)):
            if pos_king[0] >= pos_king[1]:
                return np.rot90(self.board, k=2)
            else:
                return np.rot90(np.flip(self.board, axis=0), k=1)
        elif (pos_king[0] >= math.floor(self.N_ROWS/2)) and (pos_king[1] <= (math.floor(self.N_COLS/2)-1)):
            if pos_king[0] + pos_king[1] <= self.N_COLS-1:
                return np.rot90(self.board, k=3)
            else:
                return np.flip(self.board, axis=0)
        elif (pos_king[0] <= (math.floor(self.N_ROWS/2)-1)) and (pos_king[1] <= (math.floor(self.N_COLS/2)-2)):
            if pos_king[0] > pos_king[1]:
                return np.rot90(np.flip(self.board, axis=1), k=1)
            else:
                return self.board
        else:
            return self.board


    """
        Determines the possible allowed moves from the current state of the booard.

        Returns
        -------
            critical_moves : list[tuple[tuple[int, int], tuple[int, int]]]
                List of tuples (from, start) of critical moves.

            other_moves : list[tuple[tuple[int, int], tuple[int, int]]]
                List of tuples (from, start) of the remaining moves.
                
    """
    def getMoves(self) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[tuple[int, int], tuple[int, int]]]]:
        pos_king = tuple(np.argwhere(self.board == KING)[0])
        pawn = WHITE if self.is_white_turn else BLACK

        king_moves = []
        same_king_axis_moves = []
        near_king_moves = []
        capturing_moves = []
        other_moves = []

        if self.is_white_turn:
            # If White turn, check KING moves
            for m in self.__getPawnMoves(pos_king[0], pos_king[1]):
                king_moves.append(m)

        for i in range(self.N_ROWS):
            for j in range(self.N_COLS):
                if self.board[i, j] != pawn: continue

                for start, end in self.__getPawnMoves(i,j):
                    if end[0] == pos_king[0] or end[1] == pos_king[1]:
                        same_king_axis_moves.append((start, end))
                    elif end[0] == pos_king[0]+1 or end[0] == pos_king[0]-1 or end[1] == pos_king[1]+1 or end[1] == pos_king[1]-1:
                        near_king_moves.append((start, end))
                    elif (self.isCaptured(end[0]+1, end[1], VERTICAL) or self.isCaptured(end[0]-1, end[1], VERTICAL) or
                          self.isCaptured(end[0], end[1]+1, HORIZONTAL) or self.isCaptured(end[0], end[1]-1, HORIZONTAL)):
                        capturing_moves.append((start, end))
                    else:
                        other_moves.append((start, end))
        
        return king_moves + same_king_axis_moves + near_king_moves + capturing_moves, other_moves


    def __getPawnMoves(self, i:int, j:int) -> Generator[tuple[tuple[int, int], tuple[int, int]]]:
        for direction in [RIGHT, UP, LEFT, DOWN]:
            n = self.numSteps(i, j, direction)
            for step in range(1, n+1):
                if direction == RIGHT:
                    target = (i, j + step)
                elif direction == UP:
                    target = (i - step, j)
                elif direction == LEFT:
                    target = (i, j - step)
                elif direction == DOWN:
                    target = (i + step, j)
                
                if self.isValidCell(target[0], target[1]):
                    yield ((i, j), target)

    """
        Applies a move in the board.
        It is assumed that the move is valid.

        Parameters
        ----------
            start : tuple[int, int]
                Coordinates of the starting position.

            end : tuple[int, int]
                Coordinates of the destination.
                
        Returns
        -------
            captures : list[tuple[tuple[int, int], BLACK|WHITE|KING]]
                List of the pawns captured with this move.
                Each element has format ((i, j), pawn).
                `(i, j)` are the coordinates of the captured pawn.
                `pawn` is the type of pawn captured.
                Useful to undo this move.
    """
    def applyMove(self, start:tuple[int, int], end:tuple[int, int]) -> list[tuple[tuple[int, int], BLACK|WHITE|KING]]:
        captured = []
        
        # Applies move
        self.board[end[0], end[1]] = self.board[start[0], start[1]]
        self.board[start[0], start[1]] = EMPTY

        # Checks if the adjacent pieces have been captured
        if self.isCaptured(end[0]+1,end[1], to_filter_axis=VERTICAL):
            captured.append( ((end[0]+1, end[1]), self.board[end[0]+1, end[1]]) )
            self.board[end[0]+1, end[1]] = EMPTY
        if self.isCaptured(end[0]-1,end[1], to_filter_axis=VERTICAL):
            captured.append( ((end[0]-1, end[1]), self.board[end[0]-1, end[1]]) )
            self.board[end[0]-1, end[1]] = EMPTY
        if self.isCaptured(end[0],end[1]+1, to_filter_axis=HORIZONTAL):
            captured.append( ((end[0], end[1]+1), self.board[end[0], end[1]+1]) )
            self.board[end[0], end[1]+1] = EMPTY
        if self.isCaptured(end[0],end[1]-1, to_filter_axis=HORIZONTAL):
            captured.append( ((end[0], end[1]-1), self.board[end[0], end[1]-1]) )
            self.board[end[0], end[1]-1] = EMPTY

        self.is_white_turn = not self.is_white_turn

        return captured


    """
        Reverts a move and restores captured pawns.
        It is assumed that the parameters are correct.
        
        Parameters
        ----------
            old_start : tuple[int, int]
                Old starting point of the move to revert.
                E.g., if the move was ((0, 0), (1, 0)), this parameters is (0, 0).
                
            old_end : tuple[int, int]
                Old destination of the move to revert.
                E.g., if the move was ((0, 0), (1, 0)), this parameters is (1, 0).

            captured : list[tuple[tuple[int, int], BLACK|WHITE|KING]]
                List of pawns the move to revert captured.
                They are restored.
    """
    def revertMove(self, old_start:tuple[int, int], old_end:tuple[int, int], captured:list[tuple[tuple[int, int], BLACK|WHITE|KING]]):
        # Reverts move
        self.board[old_start[0], old_start[1]] = self.board[old_end[0], old_end[1]]
        self.board[old_end[0], old_end[1]] = EMPTY
        
        # Reverts captured pawn
        for el in captured:
            pos = el[0]
            pawn = el[1]
            self.board[pos[0], pos[1]] = pawn

        self.is_white_turn = not self.is_white_turn


    """
        Determines the status of the current board.

        Returns
        -------
            game_state : BLACK_WIN | WHITE_WIN | OPEN
                The status of the board.
    """
    def getGameState(self) -> BLACK_WIN | WHITE_WIN | OPEN:
        pos_king = np.argwhere(self.board==KING)
        if len(pos_king)==0:
            return BLACK_WIN
        elif ((pos_king[0][0], pos_king[0][1]) in ESCAPE_TILES):
            return WHITE_WIN
        return OPEN  
              

    """
        Checks if a position is valid.

        Parameters
        ----------
            i, j: int
                Row and column of the position to check.

        Returns
        -------
            is_valid : bool
    """
    def isValidCell(self, i: int, j: int) -> bool:
        return (0 <= i < self.N_ROWS) and (0 <= j < self.N_COLS)

    
    """
        Checks if a cell is a wall (camp or the castle).
        Useful to determine a capture.

        Parameters
        ----------
            i, j: int
                Row and column of the position to check.

        Returns
        -------
            is_wall : bool
    """
    def isWall(self, i: int, j: int)->bool:
        return (i, j) in CAMP_DICT.keys() or (i, j) == CASTLE_TILE
    
    
    """
        Checks if there is a capture in a given position (i, j).
        Handles both king and soldiers capture.
        If not specified, captures are checked both vertically and horizontally.

        Parameters
        ----------
            i, j: int
                Row and column of the position.

            to_filter_axis : None | VERTICAL | HORIZONTAL
                If None, both vertical and horizotal axis are checked for a normal capture.
                If VERTICAL or HORIZONTAL, only that axis is checked.

        Returns
        -------
            is_captured : bool
                True if the pawn in position (i, j) has been captured.
                False otherwise.
    """
    def isCaptured(self, i: int, j: int, to_filter_axis:None|VERTICAL|HORIZONTAL=None)->bool:
        if not self.isValidCell(i, j): return False
        
        if self.board[i, j] == EMPTY:
            return False
        # King captured in castle
        elif (self.board[i, j] == KING and (i, j) == CASTLE_TILE):
            return (
                self.board[i+1, j] == BLACK and
                self.board[i-1, j] == BLACK and
                self.board[i, j+1] == BLACK and
                self.board[i, j-1] == BLACK
            )
        # King captured near castle
        elif (self.board[i, j] == KING and (i, j) in NEAR_CASTLE_TILES):
            cnt_black = 0
            for k in (+1, -1): # Coordinates increment
                if (i+k, j) != CASTLE_TILE:
                    if self.board[i+k, j] == BLACK:
                        cnt_black += 1
                if (i, j+k) != CASTLE_TILE:
                    if self.board[i, j+k] == BLACK:
                        cnt_black += 1
            if cnt_black == 3:
                return True
        # Normal capture
        else:
            is_vertically_captured = False
            is_horizontally_captured = False
            
            if to_filter_axis is None or to_filter_axis == VERTICAL:
                is_vertically_captured = (
                    self.isValidCell(i+1, j) and self.isValidCell(i-1, j) and
                    self.isCapturingElementFor(i, j, i+1, j) and self.isCapturingElementFor(i, j, i-1, j)
                )
            if to_filter_axis is None or to_filter_axis == HORIZONTAL:
                is_horizontally_captured = (
                    self.isValidCell(i, j+1) and self.isValidCell(i, j-1) and
                    self.isCapturingElementFor(i, j, i, j+1) and self.isCapturingElementFor(i, j, i, j-1)
                )

            return is_vertically_captured or is_horizontally_captured
    
    
    """
        Checks if the cell (i, j) is an obstacle (wall, pawn or board bounds).
        Handles the case of a black pawn inside its camp.

        Parameters
        ----------
            i, j: int
                Row and column of the position.

            num_camp : None|int
                Identification number of the camp if the pawn whose this call is referencing is in a camp.
                None otherwise.

        Returns
        -------
            is_obstacle : bool
                True if the position is an obstacle.
                False otherwise.
    """
    def isObstacle(self, i:int, j:int, num_camp:None|int)->bool:
        # Out of the board
        if not self.isValidCell(i, j):
            return True
            
        # Pawn on the way
        if self.board[i, j] != EMPTY:
            return True
        
        # In the case of black inside the camp, the camp in which the black is
        # is not considered an obstacle
        if (num_camp is not None) and CAMP_DICT.get((i, j), None) == num_camp:
            return False
        else:
            return self.isWall(i, j)
        
    
    """
        Determines if a cell (i, j) contains an element that can capture
        a pawn at a given position.

        Parameters
        ----------
            pawn_color : WHITE | BLACK
                The color of the pawn to check if the cell (i, j) is a capturing cell.

            pawn_i, pawn_j: int
                Row and column of the pawn to check (i.e. the one that will potentially be captured).

            check_i, check_j: int
                Row and column of the cell to check (i.e. the one that will potentially be the capturer).
        
        Returns
        -------
            is_capturing : bool
    """
    def isCapturingElementFor(self, pawn_i:int, pawn_j:int, check_i:int, check_j:int) -> bool:
        if self.board[pawn_i, pawn_j] == WHITE or self.board[pawn_i, pawn_j] == KING:
            return self.board[check_i, check_j] == BLACK or self.isWall(check_i, check_j)
        else:
            return (self.board[check_i, check_j] == WHITE or self.board[check_i, check_j] == KING or 
                (self.isWall(check_i, check_j) and (pawn_i, pawn_j) not in CAMP_DICT))


    """
        Checks if a (black) pawn is inside a camp.

        Parameters
        ----------
            i, j: int
                Row and column of the pawn.

        Returns
        -------
            camp_number : None | int
                None if the pawn is not in a camp.
                The identification number of the camp otherwise.
    """       
    def getCampOfPawnAt(self, i:int, j:int) -> None|int:
        if self.board[i,j] != BLACK:
            return None
        return CAMP_DICT.get((i,j), None)        


    """
        Determines the number of steps a pawn can do towards a direction.

        Parameters
        ----------
            i, j: int
                Row and column of the pawn.
            
            direction : RIGHT | UP | LEFT | DOWN
                Direction to check.

        Returns
        -------
            num_steps : int
                Number of steps the pawn can make.
    """
    def numSteps(self, i:int, j:int, direction:RIGHT|UP|LEFT|DOWN) -> int:
        num_camp = self.getCampOfPawnAt(i, j)
        num = 1
        if direction == RIGHT:
            while(not self.isObstacle(i, j + num, num_camp)):
                num +=1
        elif direction == UP:
            while(not self.isObstacle(i - num, j, num_camp)):
                num +=1
        elif direction == LEFT:
            while(not self.isObstacle(i, j - num, num_camp)):
                num +=1
        elif direction == DOWN:
            while(not self.isObstacle(i + num, j, num_camp)):
                num +=1
        return num - 1


    """
        Determines the score of the current configuration.

        Parameters
        ----------
            player_color : BLACK|WHITE
                For which player the score is computed.
        
            weights : list[float]
                Weights to apply to the heuristics.

        Returns
        -------
            score : float
                Score or heuristic of the board.
    """
    def evaluate(self, 
            player_color:BLACK|WHITE,
            max_depth: int,
            positive_weights:list[float],
            negative_weights:list[float],
        )->float:
        game_state = self.getGameState()
        
        if game_state == BLACK_WIN: 
           return (MAX_SCORE+max_depth) if player_color == BLACK else (MIN_SCORE-max_depth)
        elif game_state == WHITE_WIN:
            return (MAX_SCORE+max_depth) if player_color == WHITE else (MIN_SCORE-max_depth)
        else:
            return self.heuristics(
                player_color, 
                positive_weights,
                negative_weights,
            )


    """
        Computes the heuristic score for this game state.

        Parameters
        ----------
            player_color : BLACK | WHITE
                Color for which compute the heuristic.

            weights : list[float]
                Weights to apply to the heuristics.

        Returns
        -------
            score : float
    """
    def heuristics(self, 
            player_color:BLACK|WHITE, 
            positive_weights:list[float],
            negative_weights:list[float],
        ) -> float:
        if player_color == WHITE:
            return (
                (
                    positive_weights[0] * self.__pawnRatio(WHITE) + 
                    positive_weights[1] * self.__avgProximityToKingRatio(WHITE) + 
                    positive_weights[2] * self.__safenessRatio(WHITE) + 
                    positive_weights[3] * self.__minDistanceToEscapeRatio()
                ) - (
                    negative_weights[0] * self.__pawnRatio(BLACK) +
                    negative_weights[1] * self.__avgProximityToKingRatio(BLACK) + 
                    negative_weights[2] * self.__safenessRatio(BLACK) +
                    negative_weights[3] * self.__kingDangerRatio()
                )
            )
        else:
            return (
                (
                    positive_weights[0] * self.__pawnRatio(BLACK) + 
                    positive_weights[1] * self.__avgProximityToKingRatio(BLACK) + 
                    positive_weights[2] * self.__safenessRatio(BLACK) +
                    positive_weights[3] * self.__kingDangerRatio()
                ) - (
                    negative_weights[0] * self.__pawnRatio(WHITE) +
                    negative_weights[1] * self.__avgProximityToKingRatio(WHITE) + 
                    negative_weights[2] * self.__safenessRatio(WHITE) +
                    negative_weights[3] * self.__minDistanceToEscapeRatio()
                )
            )


    """
        Determines the number of pawns of a certain color (king excluded).

        Parameters
        ----------
            color : WHITE|BLACK
                Color of the pawn to check.

        Returns
        -------
            num_pawns : int
    """
    def __pawnRatio(self, color:WHITE|BLACK) -> int:
        count = np.sum(self.board == color)
        if color == WHITE: return count / self.N_WHITES
        else: return count / self.N_BLACKS


    """
        Determines the average Manhattan distance of 
        the pawns of a certain color to the king.
        Returns 1 if the pawns are near the king and 0 if they are far.

        Parameters
        ----------
            color : WHITE|BLACK
                Color of the pawn to check.

        Returns
        -------
            avg_proximity_ratio : float
    """
    def __avgProximityToKingRatio(self, color:WHITE|BLACK)->float:
        pos_king = tuple(np.argwhere(self.board == KING)[0])
        dist = []
        for i in range(self.N_ROWS):
            for j in range(self.N_COLS):
                if self.board[i,j] == color:
                    dist.append(abs(pos_king[0] - i) + abs(pos_king[1] - j))
        avg_dist = self.MAX_DIST_TO_KING if len(dist) == 0 else (sum(dist)/len(dist))
        return 1 - (avg_dist / self.MAX_DIST_TO_KING)


    """
        Determines the safeness ratio of the pawns of a certain color (king included).
        The safeness ratio is 0 if all the pawns are surrounded 
        (i.e. all pawns are missing a single opponent pawn to be captured)
        and is 1 if none of the pawns are close to opponent pawns or walls.
        
        Parameters
        ----------
            color : WHITE|BLACK
                Color of the pawn to check.

        Returns
        -------
            threat_ratio : float
    """
    def __safenessRatio(self, color:WHITE|BLACK) -> float:
        threats = 0
        total_possible_threats = 0
        for i in range(self.N_ROWS):
            for j in range(self.N_COLS):
                if (i, j) in CAMP_DICT: continue # Black pawns inside a camp are not counted

                if (self.board[i, j] == color) or (self.board[i, j] == KING and color == WHITE):
                    if self.isValidCell(i+1, j) and self.isValidCell(i-1, j): 
                        total_possible_threats += 1
                        if ((self.isCapturingElementFor(i, j, i+1, j) and self.board[i-1, j] == EMPTY) or
                            (self.isCapturingElementFor(i, j, i-1, j) and self.board[i+1, j] == EMPTY)):
                            threats += 1
                    if self.isValidCell(i, j+1) and self.isValidCell(i, j-1): 
                        total_possible_threats += 1
                        if ((self.isCapturingElementFor(i, j, i, j+1) and self.board[i, j-1] == EMPTY) or
                            (self.isCapturingElementFor(i, j, i, j-1) and self.board[i, j+1] == EMPTY)):
                            threats += 1

        if total_possible_threats == 0: return 1  
        else: return 1 - (threats / total_possible_threats)


    """
        Determines the minimum distance between the king and the free escape tiles.

        Returns
        -------
            min_distance : float
    """
    def __minDistanceToEscapeRatio(self) -> float:
        pos_king = tuple(np.argwhere(self.board == KING)[0])
        m = self.MAX_DIST_TO_ESCAPE
        for t in ESCAPE_TILES:
            if self.board[t[0], t[1]] == EMPTY:
                dist = abs(pos_king[0] - t[0]) + abs(pos_king[1] - t[1])
                if dist < m:
                    m = dist
        return 1 - (m / self.MAX_DIST_TO_ESCAPE)
    

    """
        Determines how much the king is in danger.
    """
    def __kingDangerRatio(self) -> float:
        pos_king = tuple(np.argwhere(self.board == KING)[0])
        blacks_around = 0

        if (self.isValidCell(pos_king[0]+1, pos_king[1])) and (self.board[pos_king[0]+1, pos_king[1]] == BLACK): blacks_around += 1
        if (self.isValidCell(pos_king[0]-1, pos_king[1])) and (self.board[pos_king[0]-1, pos_king[1]] == BLACK): blacks_around += 1
        if (self.isValidCell(pos_king[0], pos_king[1]+1)) and (self.board[pos_king[0], pos_king[1]+1] == BLACK): blacks_around += 1
        if (self.isValidCell(pos_king[0], pos_king[1]-1)) and (self.board[pos_king[0], pos_king[1]-1] == BLACK): blacks_around += 1

        return blacks_around / 4