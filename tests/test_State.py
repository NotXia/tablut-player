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

        b = [[E,E,E,B,B,B,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [K,E,B,E,W,E,E,E,E],
             [E,B,E,E,E,E,E,E,E],
             [B,E,E,E,E,E,W,B,B],
             [B,E,E,E,W,E,E,E,B],
             [E,E,E,E,W,E,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,B,B,B,E,E,E]]
        s = State(np.array(b, dtype=np.byte), True)
        self.assertEqual(s.numSteps(3,1,0), 6)
        self.assertEqual(s.numSteps(3,1,1), 3)
        self.assertEqual(s.numSteps(3,1,2), 0)
        self.assertEqual(s.numSteps(3,1,3), 0)

    def test_getMoves(self):
        self.assertEqual(len(state.getMoves()), 56)

    
    def test_insideCamp(self):
        b = [[E,E,E,B,B,B,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [K,E,B,E,W,E,E,E,E],
             [B,E,E,E,W,E,E,E,B],
             [B,E,E,E,E,W,W,B,B],
             [B,E,E,E,W,E,E,E,B],
             [E,E,E,E,W,E,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,B,B,B,E,E,E]]
        s = State(np.array(b, dtype=np.byte), True)
        self.assertTrue(s.insideCamp(0,3)[0])
        self.assertTrue(s.insideCamp(0,4)[0])
        self.assertTrue(s.insideCamp(0,5)[0])
        self.assertFalse(s.insideCamp(2,2)[0])
        
    def test_isTerminal(self):
        b = [[E,E,E,B,B,B,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,E,W,E,E,E,E],
             [B,E,E,E,W,E,E,E,B],
             [B,B,W,W,K,W,W,B,B],
             [B,E,E,E,W,E,E,E,B],
             [E,E,E,E,W,E,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,B,B,B,E,E,E]]
        s = State(np.array(b, dtype=np.byte), True)
        self.assertEqual(s.isTerminal(), GameState.OPEN)

        b = [[E,E,E,B,B,B,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [K,E,E,E,W,E,E,E,E],
             [B,E,E,E,W,E,E,E,B],
             [B,B,E,E,E,W,W,B,B],
             [B,E,E,E,W,E,E,E,B],
             [E,E,E,E,W,E,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,B,B,B,E,E,E]]
        s = State(np.array(b, dtype=np.byte), True)
        self.assertEqual(s.isTerminal(), GameState.WHITE_WIN)

        b = [[E,E,E,B,B,B,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,E,W,E,E,E,E],
             [E,E,E,E,W,E,E,E,B],
             [B,B,E,B,E,W,W,B,B],
             [B,E,E,E,W,E,E,E,B],
             [E,E,E,E,W,E,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,B,B,B,E,E,E]]
        s = State(np.array(b, dtype=np.byte), True)
        self.assertEqual(s.isTerminal(), GameState.BLACK_WIN)

    def test_isCaptured(self):
        b = [[E,E,E,B,B,B,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,E,E,W,E,E,E],
             [E,E,E,B,W,B,E,E,E],
             [B,B,W,W,K,W,W,B,B],
             [B,E,E,E,W,E,E,E,B],
             [E,E,E,E,W,E,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,B,B,B,E,E,E]]
        s = State(np.array(b, dtype=np.byte), True)
        self.assertTrue(s.isCaptured(3,4))
        self.assertTrue(s.isCaptured(3,5))
        self.assertFalse(s.isCaptured(4,4))
        self.assertFalse(s.isCaptured(0,0))

    


if __name__ == "__main__":
    unittest.main()
