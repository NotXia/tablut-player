import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))
from gametree.State import *
import numpy as np
import unittest

B = BLACK
W = WHITE
K = KING
E = EMPTY

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
        self.assertEqual(state.numSteps(2, 4, RIGHT), 4)
        self.assertEqual(state.numSteps(2, 4, UP), 0)
        self.assertEqual(state.numSteps(2, 4, LEFT), 4)
        self.assertEqual(state.numSteps(2, 4, DOWN), 0)

        self.assertEqual(state.numSteps(4, 4, RIGHT), 0)
        self.assertEqual(state.numSteps(4, 4, UP), 0)
        self.assertEqual(state.numSteps(4, 4, LEFT), 0)
        self.assertEqual(state.numSteps(4, 4, DOWN), 0)

        self.assertEqual(state.numSteps(4, 3, RIGHT), 0)
        self.assertEqual(state.numSteps(4, 3, UP), 3)
        self.assertEqual(state.numSteps(4, 3, LEFT), 0)
        self.assertEqual(state.numSteps(4, 3, DOWN), 3)

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
        self.assertEqual(s.numSteps(3,1,RIGHT), 6)
        self.assertEqual(s.numSteps(3,1,UP), 3)
        self.assertEqual(s.numSteps(3,1,LEFT), 0)
        self.assertEqual(s.numSteps(3,1,DOWN), 0)

    def test_getMoves(self):
        critical, others = state.getMoves()
        self.assertEqual(len(critical) + len(others), 56)

    
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
        self.assertNotEqual(s.getCampOfPawnAt(0,3), None)
        self.assertNotEqual(s.getCampOfPawnAt(0,4), None)
        self.assertNotEqual(s.getCampOfPawnAt(0,5), None)
        self.assertEqual(s.getCampOfPawnAt(2,2), None)
        self.assertEqual(s.getCampOfPawnAt(4,1), None)
        
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
        self.assertEqual(s.getGameState(), OPEN)

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
        self.assertEqual(s.getGameState(), WHITE_WIN)

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
        self.assertEqual(s.getGameState(), BLACK_WIN)

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
        self.assertFalse(s.isCaptured(3,4, to_filter_axis=VERTICAL))
        self.assertTrue(s.isCaptured(3,4, to_filter_axis=HORIZONTAL))
        self.assertTrue(s.isCaptured(3,5))
        self.assertTrue(s.isCaptured(3,5, to_filter_axis=VERTICAL))
        self.assertFalse(s.isCaptured(3,5, to_filter_axis=HORIZONTAL))
        self.assertFalse(s.isCaptured(4,4))
        self.assertFalse(s.isCaptured(4,4, to_filter_axis=VERTICAL))
        self.assertFalse(s.isCaptured(4,4, to_filter_axis=HORIZONTAL))
        self.assertFalse(s.isCaptured(0,0))
        self.assertFalse(s.isCaptured(0,0, to_filter_axis=VERTICAL))
        self.assertFalse(s.isCaptured(0,0, to_filter_axis=HORIZONTAL))

        b = [[E,E,E,B,B,B,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,E,E,W,E,E,E],
             [E,E,E,B,B,B,E,E,E],
             [B,B,W,W,K,B,W,B,B],
             [B,E,E,E,B,E,E,E,B],
             [E,E,E,E,W,E,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,B,B,B,E,E,E]]
        s = State(np.array(b, dtype=np.byte), True)
        self.assertFalse(s.isCaptured(4, 4))

        b = [[E,E,E,B,B,B,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,E,E,W,E,E,E],
             [E,E,E,B,B,B,E,E,E],
             [B,B,W,B,K,B,W,B,B],
             [B,E,E,E,B,E,E,E,B],
             [E,E,E,E,W,E,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,B,B,B,E,E,E]]
        s = State(np.array(b, dtype=np.byte), True)
        self.assertTrue(s.isCaptured(4, 4))

        b = [[E,E,E,B,B,B,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,E,E,W,E,E,E],
             [E,E,E,B,K,B,E,E,E],
             [B,B,W,W,E,B,W,B,B],
             [B,E,E,E,B,E,E,E,B],
             [E,E,E,E,W,E,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,B,B,B,E,E,E]]
        s = State(np.array(b, dtype=np.byte), True)
        self.assertFalse(s.isCaptured(3, 4))

        b = [[E,E,E,B,B,B,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,E,B,W,E,E,E],
             [E,E,E,B,K,B,E,E,E],
             [B,B,W,W,E,B,W,B,B],
             [B,E,E,E,B,E,E,E,B],
             [E,E,E,E,W,E,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,B,B,B,E,E,E]]
        s = State(np.array(b, dtype=np.byte), True)
        self.assertTrue(s.isCaptured(3, 4))

        b = [[E,E,E,B,B,B,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,E,E,W,E,E,E],
             [E,E,W,B,K,B,E,E,E],
             [B,B,W,W,E,B,W,B,B],
             [B,E,E,E,B,E,E,E,B],
             [E,E,E,E,W,E,E,E,E],
             [E,E,E,E,B,E,E,E,E],
             [E,E,E,B,B,B,E,E,E]]
        s = State(np.array(b, dtype=np.byte), True)
        self.assertTrue(s.isCaptured(3, 3, HORIZONTAL))

if __name__ == "__main__":
    unittest.main()
