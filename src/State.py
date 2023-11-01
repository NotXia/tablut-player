from __future__ import annotations
import numpy as np
import numpy.typing as npt
from copy import deepcopy

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
    def __init__(self, board: npt.NDArray[np.byte], white_turn: bool):
        self.board = board
        self.is_white_turn = white_turn

    
    def getMoves(self)->list[State]:
        res =[]
        # if self.white_turn:
        for i in range(9):
            for j in range(9):
                if (self.is_white_turn and (self.board[i, j] == WHITE or self.board[i, j] == KING)) or (not self.is_white_turn and self.board[i, j] == BLACK):
                    for direction in [RIGHT, UP, LEFT, DOWN]:
                        n = self.numSteps(i, j, direction)
                        for step in range(1, n+1):
                            new_state = State(deepcopy(self.board), not self.is_white_turn)
                            original_piece = self.board[i,j]
                            new_state.board[i, j] = EMPTY
                            if direction == RIGHT:
                                target = (i, j + step)
                            elif direction == UP:
                                target = (i - step, j)
                            elif direction == LEFT:
                                target = (i, j - step)
                            elif direction == DOWN:
                                target = (i + step, j)
                            new_state.board[target[0], target[1]] = original_piece

                            # Check if the  adjacent pieces are captured
                            try:
                                if new_state.board[target[0]+1, target[1]] != EMPTY and new_state.isCaptured(target[0]+1,target[1]):
                                    new_state.board[target[0]+1, target[1]] = EMPTY
                                if new_state.board[target[0]-1, target[1]] != EMPTY and new_state.isCaptured(target[0]-1,target[1]):
                                    new_state.board[target[0]-1, target[1]] = EMPTY
                                if new_state.board[target[0], target[1]+1] != EMPTY and new_state.isCaptured(target[0],target[1]+1):
                                    new_state.board[target[0], target[1]+1] = EMPTY
                                if new_state.board[target[0], target[1]-1] != EMPTY and new_state.isCaptured(target[0],target[1]-1):
                                    new_state.board[target[0], target[1]-1] = EMPTY
                            except IndexError:
                                pass
                            
                            res.append(new_state)
        return res
    
    def isTerminal(self):
        pos_king = np.argwhere(self.board==KING)
        if len(pos_king)==0:
            return BLACK_WIN
        elif ((pos_king[0][0], pos_king[0][1]) in ESCAPE_TILES):
            return WHITE_WIN
        return OPEN  
              
    def H(self)->float:
        raise NotImplementedError()

    def evaluate(self)->float:
        raise NotImplementedError()

    def isValidCell(self, i, j)->bool:
        return (0 <= i <= 8) and (0 <= j <= 8)

    # Check if the cell is a camp or the castle
    def isWall(self, i, j)->bool:
        return (i, j) in CAMP_DICT.keys() or (i, j) == CASTLE_TILE
    
    # Check if the pawn in position i, j is captured
    def isCaptured(self, i, j)->bool:
        if not self.isValidCell(i, j): return False
        
        if self.board[i, j] == EMPTY:
            return False
        # King captured in castle
        elif (self.board[i, j] == KING and
            (i, j) == CASTLE_TILE and
            self.board[i+1, j] == BLACK and
            self.board[i-1, j] == BLACK and
            self.board[i, j+1] == BLACK and
            self.board[i, j-1] == BLACK
            ):
            return True
        # King captured near castle
        elif (self.board[i, j] == KING and 
              (i, j) in NEAR_CASTLE_TILES):
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
            """
                C'e un errore nella normal capture nel seguente caso:
                E W E
                E B E
                E W W
                In cui il black si è mosso spontaneamente tra i 2 white e 
                dopodichè succede questo:
                E W E
                E B W
                E W E
                Che non è una cattura ma viene riconosciuta come tale perchè
                il black ha un white sopra e un white sotto
            """
            to_check = BLACK
            if self.board[i, j] == BLACK:
                to_check = WHITE
            
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
    
    def isObstacle(self, i, j, is_black_inside, num_camp)->bool:
        # Out of the board
        if not self.isValidCell(i, j):
            return True
            
        # Pawn on the way
        if self.board[i, j] != EMPTY:
            return True
        
        # In the case of black inside the camp, the camp in which the black is
        # is not considered an obstacle
        if is_black_inside and CAMP_DICT[(i, j)] == num_camp:
            return False
        else:
            return self.isWall(i, j)
            
    # Return the number of steps that a piece can do in a direction
    
    def numSteps(self, i, j, direction):
        is_black_inside, num_camp = self.isInsideCamp(i, j)
        num = 1
        if direction == RIGHT:
            while(not self.isObstacle(i, j + num, is_black_inside, num_camp)):
                num +=1
        elif direction == UP:
            while(not self.isObstacle(i - num, j, is_black_inside, num_camp)):
                num +=1
        elif direction == LEFT:
            while(not self.isObstacle(i, j - num, is_black_inside, num_camp)):
                num +=1
        elif direction == DOWN:
            while(not self.isObstacle(i + num, j, is_black_inside, num_camp)):
                num +=1
        return num - 1

    # Checks if Black is inside the camp
    # Returns True and the number of the camp if it is inside
    # Returns False and None otherwise          
    def isInsideCamp(self, i, j):
        if self.board[i,j] != BLACK:
            return False, None
        
        if ((i == 0 and 3 <= j <= 5) or (i == 1 and j == 4) or
            (i == 8 and 3 <= j <= 5) or (i == 7 and j == 4) or
            (j == 0 and 3 <= i <= 5) or (i == 4 and j == 1) or
            (j == 8 and 3 <= i <= 5) or (i == 4 and j == 7)):
            return True, CAMP_DICT[(i,j)]
        
        return False, None