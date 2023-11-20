from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Generator
import cython
import logging
logger = logging.getLogger(__name__)
if not cython.compiled: logger.warn(f"Using non-compiled {__file__} module")


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
        else:
            raise ValueError("Unknown rules")


    def __str__(self):
        return f"WhiteTurn = {self.is_white_turn}\n {str(self.board)}"

    
    """
        Determines the possible allowed moves from the current state of the booard.

        Returns
        -------
            new_moves : Generator[tuple[tuple[int, int], tuple[int, int]]]
                Generator that returns a tuple (from, start).
                `from` and `start` are coordinates (i, j).
    """
    def getMoves(self) -> Generator[tuple[tuple[int, int], tuple[int, int]]]:
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
            pawn_color = WHITE
            if self.board[i, j] == BLACK:
                pawn_color = BLACK

            is_vertically_captured = False
            is_horizontally_captured = False
            
            if to_filter_axis is None or to_filter_axis == VERTICAL:
                is_vertically_captured = (
                    self.isValidCell(i+1, j) and self.isValidCell(i-1, j) and
                    self.isCapturingElementFor(pawn_color, i+1, j) and self.isCapturingElementFor(pawn_color, i-1, j)
                )
            if to_filter_axis is None or to_filter_axis == HORIZONTAL:
                is_horizontally_captured = (
                    self.isValidCell(i, j+1) and self.isValidCell(i, j-1) and
                    self.isCapturingElementFor(pawn_color, i, j+1) and self.isCapturingElementFor(pawn_color, i, j-1)
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
        a pawn of a given color.

        Parameters
        ----------
            pawn_color : WHITE | BLACK
                The color of the pawn to check if the cell (i, j) is a capturing cell.

            i, j: int
                Row and column to check.
        
        Returns
        -------
            is_capturing : bool
    """
    def isCapturingElementFor(self, pawn_color:WHITE|BLACK, i:int, j:int) -> bool:
        if pawn_color == WHITE:
            return self.board[i, j] == BLACK or self.isWall(i, j)
        else:
            return self.board[i, j] == WHITE or self.board[i, j] == KING or self.isWall(i, j)


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
        
        Returns
        -------
            score : float
                Score or heuristic of the board.
    """
    def evaluate(self, player_color:BLACK|WHITE)->float:
        game_state = self.getGameState()
        
        if game_state == BLACK_WIN: 
           return MAX_SCORE if player_color == BLACK else MIN_SCORE
        elif game_state == WHITE_WIN:
            return MAX_SCORE if player_color == WHITE else MIN_SCORE
        else:
            return self.heuristics(player_color)


    """
        Computes the heuristic score for this game state.

        Parameters
        ----------
            player_color : BLACK | WHITE
                Color for which compute the heuristic.

        Returns
        -------
            score : float
    """
    def heuristics(self, player_color:BLACK|WHITE) -> float:
        if player_color == WHITE:
            return (
                self.__countPawn(WHITE)/self.N_WHITES + 
                -self.__countPawn(BLACK)/self.N_BLACKS +
                # -self.__avgDistanceToKing(WHITE) + 
                # self.__avgDistanceToKing(BLACK) + 
                # -self.__threatRatio(WHITE) +
                # self.__threatRatio(BLACK) +
                # self.__minDistanceToEscape() +
                0
            )
        else:
            return (
                self.__countPawn(BLACK)/self.N_BLACKS + 
                -self.__countPawn(WHITE)/self.N_WHITES +
                # -self.__avgDistanceToKing(BLACK) + 
                # self.__avgDistanceToKing(WHITE) + 
                # -self.__threatRatio(BLACK) +
                # self.__threatRatio(WHITE) +
                # -self.__minDistanceToEscape() +
                0
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
    def __countPawn(self, color:WHITE|BLACK) -> int:
        return np.sum(self.board == color)


    """
        Determines the average Manhattan distance of 
        the pawns of a certain color to the king.

        Parameters
        ----------
            color : WHITE|BLACK
                Color of the pawn to check.

        Returns
        -------
            avg_distance : float
    """
    def __avgDistanceToKing(self, color:WHITE|BLACK)->float:
        # TODO What if no whites remaining
        pos_king = tuple(np.argwhere(self.board == KING)[0])
        dist = []
        for i in range(self.N_ROWS):
            for j in range(self.N_COLS):
                if self.board[i,j] == color:
                    dist.append(abs(pos_king[0] - i) + abs(pos_king[1] - j))
        return sum(dist)/len(dist)


    """
        Determines the threat ratio of the pawns of a certain color (king included).
        The surrounding ratio is 1 if all the pawns are surrounded 
        (i.e. any pawn is missing a single opponent pawn to be captured)
        and is 0 if none of the pawns are close to opponent pawns or walls.
        
        Parameters
        ----------
            color : WHITE|BLACK
                Color of the pawn to check.

        Returns
        -------
            threat_ratio : float
    """
    def __threatRatio(self, color:WHITE|BLACK) -> float:
        threats = 0
        total_possible_threats = 0
        for i in range(self.N_ROWS):
            for j in range(self.N_COLS):
                if (i, j) in CAMP_DICT: continue # Black pawns inside a camp are not counted

                if (self.board[i, j] == color) or (self.board[i, j] == KING and color == WHITE):
                    if self.isValidCell(i+1, j) and self.isValidCell(i-1, j): 
                        total_possible_threats += 1
                        if ((self.isCapturingElementFor(color, i+1, j) and self.board[i-1, j] == EMPTY) or
                            (self.isCapturingElementFor(color, i-1, j) and self.board[i+1, j] == EMPTY)):
                            threats += 1
                    if self.isValidCell(i, j+1) and self.isValidCell(i, j-1): 
                        total_possible_threats += 1
                        if ((self.isCapturingElementFor(color, i, j+1) and self.board[i, j-1] == EMPTY) or
                            (self.isCapturingElementFor(color, i, j-1) and self.board[i, j+1] == EMPTY)):
                            threats += 1

        return 0 if total_possible_threats == 0 else (threats / total_possible_threats)


    """
        Determines the minimum distance between the king and the free escape tiles.

        Returns
        -------
            min_distance : float
    """
    def __minDistanceToEscape(self) -> float:
        # TODO What if none are free
        pos_king = tuple(np.argwhere(self.board == KING)[0])
        m = 100
        for t in ESCAPE_TILES:
            if self.board[t[0], t[1]] == EMPTY:
                dist = abs(pos_king[0] - t[0]) + abs(pos_king[1] - t[1])
                if dist < m:
                    m = dist
        return m          