from .State import State, OPEN, WHITE, BLACK, MAX_SCORE, MIN_SCORE
import numpy as np
from .TreeNode import TreeNode
import time
import cython
import logging
logger = logging.getLogger(__name__)
if not cython.compiled: logger.warn(f"Using non-compiled {__file__} module")

"""
    Class that represents the whole game tree.
"""
class Tree():
    def __init__(self, initial_state, player_color, debug=False):
        self.state: State = initial_state
        self.player_color = player_color
        self.root = TreeNode(None, None)

        self.__debug = debug
        if self.__debug:
            self.__explored_nodes = 0


    """
        Determines the next best move.

        Parameters
        ----------
            timeout : float
                Seconds available to make a choice.

        Returns
        -------
            from : tuple[int, int]
                Starting coordinates of the move.

            to : tuple[int, int]
                Ending coordinates of the move.
                
            best_score : float
                Score of the chosen move.
    """
    def decide(self, timeout):
        if self.__debug: 
            self.__explored_nodes = 0
        
        end_timestamp = time.time() + timeout
        best_child = None
        depth = 0
        
        if self.root.score == MAX_SCORE:
            # A winning move is already known, minimax is not necessary.
            # This also prevents possible loops.
            logger.debug("Following winning path")
            best_score = MAX_SCORE
            
            for child in self.root.children:    
                if child.score == best_score:
                    best_child = child
                    break
        else:
            while time.time() < end_timestamp:
                depth += 1
                curr_best_score = self.minimax(self.root, depth, -np.inf, +np.inf, end_timestamp)
                if curr_best_score is None:
                    depth -= 1
                    break
                best_score = curr_best_score

                for child in self.root.children:
                    if child.score == best_score:
                        best_child = child
                        break
        
        if self.__debug:
            logger.debug(f"Explored depth = {depth}")
            logger.debug(f"Explored nodes: {self.__explored_nodes}, {self.__explored_nodes/(timeout):.2f} nodes/s")

        
        self.root = best_child
        _ = self.state.applyMove(best_child.start, best_child.end)
        return best_child.start, best_child.end, best_score
    

    """
        Moves the root of the tree to the node containing the opponent's move.
        If it does not exist, the tree is resetted.

        Parameters
        ----------
            next_state : State
                State of the board after the opponent's move.
    """
    def applyOpponentMove(self, next_state: State):
        try:
            for child in self.root.children:
                captured = self.state.applyMove(child.start, child.end)
                if np.all(self.state.board == next_state.board):
                    logger.debug("Not dropping tree")
                    # Move found, update the root and
                    # leave the board status as is (do not need to revert).
                    self.root = child
                    return
                self.state.revertMove(child.start, child.end, captured)
        except:
            logger.error("Error in applying opponent move")
        
        # Move not found among the children of the root,
        # the current tree is deleted.
        logger.debug("Dropping tree")
        self.state = next_state
        self.root = TreeNode(None, None)


    """
        Runs minimax with alpha-beta pruning on a given node.

        Parameters
        ----------
            tree_node : TreeNode
                Node to explore.

            max_depth : int
                Maximum reachable depth before evaluating the node.

            alpha, beta : float
                Alpha and beta for pruning

        Returns
        -------
            best_score : float
                Best children's score found.
    """
    def minimax(self, 
        tree_node:TreeNode, 
        max_depth:int, 
        alpha:float, beta:float, 
        timeout_timestamp:float) -> tuple[float|None, TreeNode|None]:
        if self.__debug:
            self.__explored_nodes += 1

        if self.state.getGameState() != OPEN or max_depth == 0:
            eval = self.state.evaluate(self.player_color)
        else:
            if ((self.state.is_white_turn and self.player_color == WHITE) or
                (not self.state.is_white_turn and self.player_color == BLACK)):
                # Max
                eval = -np.inf
                for child in tree_node.getChildren(self.state):
                    if time.time() >= timeout_timestamp: return None # Timeout
                    
                    captured = self.state.applyMove(child.start, child.end)
                    eval_minimax = self.minimax(child, max_depth-1, alpha, beta, timeout_timestamp)
                    self.state.revertMove(child.start, child.end, captured)
                    
                    if eval_minimax is None: return None # Timeout
                    
                    eval = max(eval, eval_minimax)
                    alpha = max(eval, alpha)
                    if eval >= beta: break # cutoff
            else:
                # Min
                eval = np.inf
                for child in tree_node.getChildren(self.state):
                    if time.time() >= timeout_timestamp: return None # Timeout

                    captured = self.state.applyMove(child.start, child.end)
                    eval_minimax = self.minimax(child, max_depth-1, alpha, beta, timeout_timestamp)
                    self.state.revertMove(child.start, child.end, captured)
                    
                    if eval_minimax is None: return None # Timeout

                    eval = min(eval, eval_minimax)
                    beta = min(eval, beta)
                    if eval <= alpha: break # cutoff

        tree_node.score = eval
        return eval
        
