from __future__ import annotations
import numpy as np
import numpy.typing as npt
from copy import deepcopy
from enum import IntEnum

class BoardCell(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    KING  = 3
class GameState(IntEnum):
    WHITE_WIN = 4
    BLACK_WIN = 5
    OPEN =  6

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

"""
    Simple class to represent the state of the game.
    The state is represented by a matrix of bytes of dimensions 9x9.
"""
class State():
    def __init__(self, board: npt.NDArray[np.byte], white_turn: bool):
        self.board = board
        self.white_turn = white_turn

    def getMoves(self)->list[State]:
        res =[]
        # if self.white_turn:
        for i in range(9):
            for j in range(9):
                if (self.white_turn and (self.board[i, j] == BoardCell.WHITE or self.board[i, j] == BoardCell.KING)) or (not self.white_turn and self.board[i, j] == BoardCell.BLACK):
                    for direction in range(4):
                        n = self.numSteps(i, j, direction)
                        for step in range(1, n+1):
                            new_state = State(deepcopy(self.board), not self.white_turn)
                            new_state.board[i, j] = BoardCell.EMPTY
                            if direction == 0:
                                target = (i, j + step)
                            elif direction == 1:
                                target = (i - step, j)
                            elif direction == 2:
                                target = (i, j - step)
                            elif direction == 3:
                                target = (i + step, j)
                            new_state.board[target[0], target[1]] = BoardCell.WHITE

                            # Check if the  adjacent pieces are captured
                            if new_state.isCaptured(target[0]+1,target[1]):
                                new_state.board[target[0]+1, target[1]] = BoardCell.EMPTY
                            if new_state.isCaptured(target[0]-1,target[1]):
                                new_state.board[target[0]-1, target[1]] = BoardCell.EMPTY
                            if new_state.isCaptured(target[0],target[1]+1):
                                new_state.board[target[0], target[1]+1] = BoardCell.EMPTY
                            if new_state.isCaptured(target[0],target[1]-1):
                                new_state.board[target[0], target[1]-1] = BoardCell.EMPTY
                            
                            res.append(new_state)
        return res
    
    def isTerminal(self)->GameState:
        pos_king = np.argwhere(self.board==BoardCell.KING)
        if len(pos_king)==0:
            return GameState.BLACK_WIN
        elif ((pos_king[0][0], pos_king[0][1]) in ESCAPE_TILES):
            return GameState.WHITE_WIN
        return GameState.OPEN  
        
          
    def H(self)->float:
        raise NotImplementedError()

    def evaluate(self)->float:
        raise NotImplementedError()

    def isValidCell(self, i, j)->bool:
        return (0 <= i <= 8) and (0 <= j <= 8)

    # Check if the cell is a camp or the castle
    def isWall(self, i, j)->bool:
        return (i, j) in [(0, 3), (0, 4), (0, 5), (1, 4),
                                    (3, 0), (4, 0), (5, 0), (4, 1),
                                    (3, 8), (4, 8), (5, 8), (4, 7),
                                    (8, 3), (8, 4), (8, 5), (7, 4),
                                    (4, 4)]

    # Check if the pawn in position i, j is captured
    def isCaptured(self, i, j)->bool:
        if not self.isValidCell(i, j): return False
        
        if self.board[i, j] == BoardCell.EMPTY:
            return False
        # King captured in castle
        elif (self.board[i, j] == BoardCell.KING and 
            self.board[i+1, j] == BoardCell.BLACK and
            self.board[i-1, j] == BoardCell.BLACK and
            self.board[i, j+1] == BoardCell.BLACK and
            self.board[i, j-1] == BoardCell.BLACK
            ):
            return True
        # King captured near castle
        elif (self.board[i, j] == BoardCell.KING and 
              (i, j) in [(3, 4), (5, 4), (4, 3), (4, 5)]):
                cnt_black = 0
                for k in (+1, -1): # Coordinates increment
                    if (i+k, j) != (4, 4):
                        if self.board[i+k, j] == BoardCell.BLACK:
                            cnt_black += 1
                    if (i, j+k) != (4, 4):
                        if self.board[i, j+k] == BoardCell.BLACK:
                            cnt_black += 1
                if cnt_black == 3:
                    return True
        # Normal capture
        else:
            to_check = BoardCell.BLACK
            if self.board[i, j] == BoardCell.BLACK:
                to_check = BoardCell.WHITE
            
            return (
                (
                    self.isValidCell(i+1, j) and self.isValidCell(i-1, j) and
                    (self.board[i+1, j] == to_check or self.isWall(i+1, j)) and 
                    (self.board[i-1, j] == to_check or self.isWall(i-1, j))
                )
                or
                (
                    self.isValidCell(i, j+1) and self.isValidCell(i, j-1) and
                    (self.board[i, j+1] == to_check or self.isWall(i, j+1)) and
                    (self.board[i, j-1] == to_check or self.isWall(i, j-1))
                )
            ) 
    
    # Check if the cell i,j is an obstacle
    # Considering also the case of black inside the camp
    def isObstacle(self, i, j, black_inside, num_camp)->bool:
        # Out of the board
        if not self.isValidCell(i, j):
            return True
            
        #Pedone vicino
        if self.board[i, j] == BoardCell.WHITE or self.board[i, j] == BoardCell.KING or self.board[i, j] == BoardCell.BLACK:
            return True
        
        if not black_inside:
            return self.isWall(i, j)
        else:
            # In the case of black inside the camp, the camp in which the black is
            # is not considered an obstacle
            if CAMP_DICT[(i, j)] == num_camp:
                return False
            else:
                return self.isWall(i, j)
            
    """
        directions in a goniometric cirlce:
        0 -> right
        1 -> up
        2 -> left
        3 -> down
    """
    # Return the number of steps that a piece can do in a direction
    def numSteps(self, i, j, direction):
        black_inside, num_camp = self.insideCamp(i, j)
        num = 1
        if direction == 0:
            while(not self.isObstacle(i, j + num, black_inside, num_camp)):
                num +=1
        elif direction == 1:
            while(not self.isObstacle(i - num, j, black_inside, num_camp)):
                num +=1
        elif direction == 2:
            while(not self.isObstacle(i, j - num, black_inside, num_camp)):
                num +=1
        elif direction == 3:
            while(not self.isObstacle(i + num, j, black_inside, num_camp)):
                num +=1
        return num - 1

    # Checks if Black is inside the camp
    # Returns True and the number of the camp if it is inside
    # Returns False and None otherwise          
    def insideCamp(self, i, j):
        if self.board[i,j] != BoardCell.BLACK:
            return False, None
        
        if i == 0 and j >= 3 and j <= 5:
            return True, CAMP_DICT[(i,j)]
        elif i == 1 and j == 4:
            return True, CAMP_DICT[(i,j)]
        
        if i == 8 and j >= 3 and j <= 5:
            return True, CAMP_DICT[(i,j)]
        elif i == 7 and j == 4:
            return True, CAMP_DICT[(i,j)]
        
        if j == 0 and i >= 3 and i <= 5:
            return True, CAMP_DICT[(i,j)]
        elif j == 1 and i == 4: 
            return True, CAMP_DICT[(i,j)]
        
        if j == 8 and i >= 3 and i <= 5:
            return True, CAMP_DICT[(i,j)]
        elif j == 7 and i == 4:
            return True, CAMP_DICT[(i,j)]
        
        return False, None