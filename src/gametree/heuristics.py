from .State import State, KING, BLACK, WHITE, ESCAPE_TILES
import numpy as np


def countPawn(state: State, color:WHITE|BLACK):
    return state.board.count(color)

"""
    Average Manhattan distance of the pawns to the king
"""
def avgDistanceToKing(state: State, color:WHITE|BLACK)->float:
    pos_king = tuple(np.argwhere(state.board == KING)[0])
    dist = []
    for i in range(9):
        for j in range(9):
            if state.board[i,j] == color:
                dist.append(abs(pos_king[0] - i) + abs(pos_king[1] - j))
    return sum(dist)/len(dist)


def countSurroundingRatio(state:State, color:WHITE|BLACK) -> float:
    opponent_color = WHITE if color == BLACK else BLACK
    surrounded = 0
    total_possible_surrounding = 0

    for i in range(9):
        for j in range(9):
            if (state.board[i, j] == color) or (state.board[i, j] == KING and color == WHITE):
                if state.isValidCell(i+1, j): 
                    total_possible_surrounding += 1
                    surrounded += 1 if state.board[i+1, j] == opponent_color or state.isWall(i+1, j) else 0
                if state.isValidCell(i-1, j): 
                    total_possible_surrounding += 1
                    surrounded += 1 if state.board[i-1, j] == opponent_color or state.isWall(i-1, j) else 0
                if state.isValidCell(i, j+1): 
                    total_possible_surrounding += 1
                    surrounded += 1 if state.board[i, j+1] == opponent_color or state.isWall(i, j+1) else 0
                if state.isValidCell(i, j-1): 
                    total_possible_surrounding += 1
                    surrounded += 1 if state.board[i, j-1] == opponent_color or state.isWall(i, j-1) else 0

    return surrounded / total_possible_surrounding


def minDistanceToEscape(state: State) -> float:
    pos_king = tuple(np.argwhere(state.board == KING)[0])
    m = 100
    for t in ESCAPE_TILES:
        dist = abs(pos_king[0] - t[0]) + abs(pos_king[1] - t[1])
        if dist < m:
            m = dist
    return m          