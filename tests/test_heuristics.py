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
     def test_countPawns(self):
          s = State(initial_state, True)
          self.assertEqual(s._State__countPawn(WHITE), s.N_WHITES)
          self.assertEqual(s._State__countPawn(BLACK), s.N_BLACKS)

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
          self.assertEqual(s._State__countPawn(WHITE), 2)
          self.assertEqual(s._State__countPawn(BLACK), 2)


     def test_avgDistanceToKing(self):
          s = State(initial_state, True)
          self.assertEqual(s._State__avgDistanceToKing(WHITE), (1+2)/2)
          self.assertEqual(s._State__avgDistanceToKing(BLACK), (3+4+5+5)/4)

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
          self.assertEqual(s._State__avgDistanceToKing(BLACK), 14.0)

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
          self.assertEqual(s._State__avgDistanceToKing(BLACK), (14+1)/2)


     def test_threatRatio(self):
          s = State(initial_state, True)
          self.assertEqual(s._State__threatRatio(WHITE), 0)
          self.assertEqual(s._State__threatRatio(BLACK), 0)

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
          self.assertEqual(s._State__threatRatio(WHITE), 1.0)
          self.assertEqual(s._State__threatRatio(BLACK), 1.0)

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
          self.assertEqual(s._State__threatRatio(BLACK), 0)

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
          self.assertEqual(s._State__threatRatio(WHITE), 0.5)
          self.assertEqual(s._State__threatRatio(BLACK), 0.5)

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
          self.assertEqual(s._State__threatRatio(WHITE), 0.5)
          self.assertEqual(s._State__threatRatio(BLACK), 0.5)

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
          self.assertEqual(s._State__threatRatio(WHITE), 1)
          self.assertEqual(s._State__threatRatio(BLACK), 0.5)


     def test_minDistanceToEscape(self):
          s = State(initial_state, True)
          self.assertEqual(s._State__minDistanceToEscape(), 6)

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
          self.assertEqual(s._State__minDistanceToEscape(), 1)

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
          self.assertEqual(s._State__minDistanceToEscape(), 5)

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
          self.assertEqual(s._State__minDistanceToEscape(), 13)


if __name__ == "__main__":
    unittest.main()
