import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))
from State import State, BLACK, WHITE, KING, EMPTY
import numpy as np

B = BLACK
W = WHITE
K = KING
E = EMPTY

initial_state =  [[E,E,E,B,B,B,E,E,E],
                  [E,E,E,E,B,E,E,E,E],
                  [E,E,E,E,W,E,E,E,E],
                  [B,E,E,E,E,E,E,E,E],
                  [B,B,W,W,K,W,W,B,B],
                  [B,E,E,E,W,E,E,E,B],
                  [E,E,E,E,W,E,E,E,E],
                  [E,E,E,E,B,E,E,E,E],
                  [E,E,E,B,B,B,E,E,E]]

board = np.array(initial_state, dtype=np.byte)
state = State(board, True)

@profile
def test_getMoves():
    for i in range(1000):
        state.getMoves()

test_getMoves()