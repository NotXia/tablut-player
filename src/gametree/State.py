from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Generator
import random
from .heuristics import *
import cython
import logging
logger = logging.getLogger(__name__)
if not cython.compiled: logger.warn(f"Using non-compiled {__file__} module")


MAX_SCORE = 1
MIN_SCORE = -1

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
    def __init__(self, board: npt.NDArray[np.byte], is_white_turn: bool):
        self.board = board
        self.is_white_turn = is_white_turn

    
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
        visited = set()
        pawn = WHITE if self.is_white_turn else BLACK
        
        if self.is_white_turn:
            # If White turn, check KING moves
            for m in self.__getPawnMoves(pos_king[0], pos_king[1]):
                yield m
        
        # Then iterate over axes near the KING,
        # Check moves for pawns in rows i = pos_king[0] +- 1
        for i in [pos_king[0], pos_king[0] + 1, pos_king[0] - 1]:
            if not self.isValidCell(i, 0): continue # Checks if the row is valid
            for j in range(9):
                if (i, j) != pos_king and self.board[i, j] == pawn:
                    visited.add((i, j))
                    for m in self.__getPawnMoves(i,j):
                        yield m

        for curr_range in [range(0, pos_king[0]-1), range(pos_king[0]+2, 9)]:
            for i in curr_range:
                for j in [pos_king[1], pos_king[1] + 1, pos_king[1] - 1]:
                    if not self.isValidCell(0, j): continue # Checks if the column is valid
                    if self.board[i, j] == pawn:
                        visited.add((i,j))
                        for m in self.__getPawnMoves(i,j):
                            yield m

        # Other pawns
        for i in range(9):
            for j in range(9):
                if self.board[i, j] == pawn and (i, j) not in visited:
                    for m in self.__getPawnMoves(i,j):
                        yield m 


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
        return (0 <= i <= 8) and (0 <= j <= 8)

    
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
            capturing_pawn = BLACK
            if self.board[i, j] == BLACK:
                capturing_pawn = WHITE

            is_vertically_captured = False
            is_horizontally_captured = False
            
            if to_filter_axis is None or to_filter_axis == VERTICAL:
                is_vertically_captured = (
                    self.isValidCell(i+1, j) and self.isValidCell(i-1, j) and
                    (self.board[i+1, j] == capturing_pawn or self.isWall(i+1, j)) and 
                    (self.board[i-1, j] == capturing_pawn or self.isWall(i-1, j))
                )
            if to_filter_axis is None or to_filter_axis == HORIZONTAL:
                is_horizontally_captured = (
                    self.isValidCell(i, j+1) and self.isValidCell(i, j-1) and
                    (self.board[i, j+1] == capturing_pawn or self.isWall(i, j+1)) and
                    (self.board[i, j-1] == capturing_pawn or self.isWall(i, j-1))
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
        
        if ((i == 0 and 3 <= j <= 5) or (i == 1 and j == 4) or
            (i == 8 and 3 <= j <= 5) or (i == 7 and j == 4) or
            (j == 0 and 3 <= i <= 5) or (i == 4 and j == 1) or
            (j == 8 and 3 <= i <= 5) or (i == 4 and j == 7)):
            return CAMP_DICT[(i,j)]
        
        return None


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
    

    def heuristics(self, player_color:BLACK|WHITE)->float:
        return random.random()


    def __str__(self):
        return f"WhiteTurn = {self.is_white_turn}\n {str(self.board)}"

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
