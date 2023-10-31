from State import State, BoardCell, GameState
import numpy as np
import unittest

B = BoardCell.BLACK
W = BoardCell.WHITE
K = BoardCell.KING
E = BoardCell.EMPTY

initial_state =  [[E,E,E,B,B,B,E,E,E],
                  [E,E,E,E,B,E,E,E,E],
                  [E,E,E,E,W,E,E,E,E],
                  [B,E,E,E,W,E,E,E,B],
                  [B,B,W,W,K,W,W,B,B],
                  [B,E,E,E,W,E,E,E,B],
                  [E,E,E,E,W,E,E,E,E],
                  [E,E,E,E,B,E,E,E,E],
                  [E,E,E,B,B,B,E,E,E]]

board = np.array(initial_state, dtype=np.byte)
state = State(board, True)

class TestState(unittest.TestCase):
    def test_numSteps(self):
        self.assertEqual(state.numSteps(2, 4, 0), 4)
        self.assertEqual(state.numSteps(2, 4, 1), 0)
        self.assertEqual(state.numSteps(2, 4, 2), 4)
        self.assertEqual(state.numSteps(2, 4, 3), 0)

        self.assertEqual(state.numSteps(4, 4, 0), 0)
        self.assertEqual(state.numSteps(4, 4, 1), 0)
        self.assertEqual(state.numSteps(4, 4, 2), 0)
        self.assertEqual(state.numSteps(4, 4, 3), 0)

        self.assertEqual(state.numSteps(4, 3, 0), 0)
        self.assertEqual(state.numSteps(4, 3, 1), 3)
        self.assertEqual(state.numSteps(4, 3, 2), 0)
        self.assertEqual(state.numSteps(4, 3, 3), 3)

    def test_getMoves(self):
        self.assertEqual(len(state.getMoves()), 56)


if __name__ == "__main__":
    unittest.main()
