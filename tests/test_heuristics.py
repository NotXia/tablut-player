import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))
from gametree.State import *
import numpy as np
import unittest

B = BLACK
W = WHITE
K = KING
E = EMPTY

initial_state =  np.array(
     [[E,E,E,B,B,B,E,E,E],
      [E,E,E,E,B,E,E,E,E],
      [E,E,E,E,W,E,E,E,E],
      [B,E,E,E,W,E,E,E,B],
      [B,B,W,W,K,W,W,B,B],
      [B,E,E,E,W,E,E,E,B],
      [E,E,E,E,W,E,E,E,E],
      [E,E,E,E,B,E,E,E,E],
      [E,E,E,B,B,B,E,E,E]], dtype=np.byte)

class TestState(unittest.TestCase):
     def test_pawnRatio(self):
          s = State(initial_state, True)
          self.assertEqual(s._State__pawnRatio(WHITE), 1)
          self.assertEqual(s._State__pawnRatio(BLACK), 1)

          s = State(np.array(
               [[E,E,E,E,E,E,E,E,E],
                [E,E,B,E,E,E,W,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,W,E,E,E,B,E,E],
                [E,E,E,E,E,E,E,E,E]], dtype=np.byte), True)
          self.assertEqual(s._State__pawnRatio(WHITE), 2/s.N_WHITES)
          self.assertEqual(s._State__pawnRatio(BLACK), 2/s.N_BLACKS)


     def test_avgProximityToKingRatio(self):
          s = State(initial_state, True)
          self.assertEqual(s._State__avgProximityToKingRatio(WHITE), 1 - (((1+2)/2) / s.MAX_DIST_TO_KING))
          self.assertEqual(s._State__avgProximityToKingRatio(BLACK), 1 - (((3+4+5+5)/4) / s.MAX_DIST_TO_KING))

          s = State(np.array(
               [[E,E,E,E,E,E,E,E,E],
                [E,K,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,B]], dtype=np.byte), True)
          self.assertEqual(s._State__avgProximityToKingRatio(BLACK), 0)

          s = State(np.array(
               [[E,E,E,E,E,E,E,E,E],
                [B,K,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,B]], dtype=np.byte), True)
          self.assertEqual(s._State__avgProximityToKingRatio(BLACK), 1 - (((14+1)/2) / s.MAX_DIST_TO_KING))


     def test_safenessRatio(self):
          s = State(initial_state, True)
          self.assertEqual(s._State__safenessRatio(WHITE), 1)
          self.assertEqual(s._State__safenessRatio(BLACK), 1)

          # Pawns near camp wall
          s = State(np.array(
               [[E,E,E,E,E,E,E,E,E],
                [E,E,E,W,E,B,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,W,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,B,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E]], dtype=np.byte), True)
          self.assertEqual(s._State__safenessRatio(WHITE), 0)
          self.assertEqual(s._State__safenessRatio(BLACK), 0)

          # Black inside camp
          s = State(np.array(
               [[E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,B,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E]], dtype=np.byte), True)
          self.assertEqual(s._State__safenessRatio(BLACK), 1)

          # Pawns near castle
          s = State(np.array(
               [[E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,W,E,B,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E]], dtype=np.byte), True)
          self.assertEqual(s._State__safenessRatio(WHITE), 0.5)
          self.assertEqual(s._State__safenessRatio(BLACK), 0.5)

          # Pawns near other pawns
          s = State(np.array(
               [[E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,W,B,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E]], dtype=np.byte), True)
          self.assertEqual(s._State__safenessRatio(WHITE), 0.5)
          self.assertEqual(s._State__safenessRatio(BLACK), 0.5)

          s = State(np.array(
               [[E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,B,E,E,E,E,E,E],
                [E,E,W,B,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E]], dtype=np.byte), True)
          self.assertEqual(s._State__safenessRatio(WHITE), 0)
          self.assertEqual(s._State__safenessRatio(BLACK), 0.5)


     def test_minDistanceToEscapeRatio(self):
          s = State(initial_state, True)
          self.assertEqual(s._State__minDistanceToEscapeRatio(), 6/s.MAX_DIST_TO_ESCAPE)

          s = State(np.array(
               [[E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,K,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E]], dtype=np.byte), True)
          self.assertEqual(s._State__minDistanceToEscapeRatio(), 1/s.MAX_DIST_TO_ESCAPE)

          s = State(np.array(
               [[E,W,W,E,E,E,E,E,E],
                [W,E,E,E,E,E,E,E,E],
                [W,K,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E]], dtype=np.byte), True)
          self.assertEqual(s._State__minDistanceToEscapeRatio(), 5/s.MAX_DIST_TO_ESCAPE)

          s = State(np.array(
               [[E,W,W,E,E,E,W,W,E],
                [W,K,E,E,E,E,E,E,W],
                [W,E,E,E,E,E,E,E,W],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [E,E,E,E,E,E,E,E,E],
                [W,E,E,E,E,E,E,E,W],
                [W,E,E,E,E,E,E,E,W],
                [E,W,W,E,E,E,W,E,E]], dtype=np.byte), True)
          self.assertEqual(s._State__minDistanceToEscapeRatio(), 13/s.MAX_DIST_TO_ESCAPE)


if __name__ == "__main__":
    unittest.main()
