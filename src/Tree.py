from State import State, OPEN, WHITE, BLACK, MAX_SCORE, MIN_SCORE
import numpy as np
from TreeNode import TreeNode
import logging
import time


"""
    Class that represents the whole game tree.
"""
class Tree():
    def __init__(self, initial_state, player_color, debug=False):
        self.state: State = initial_state
        self.player_color = player_color
        self.root = TreeNode(None, None, None)

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
        
        if self.root.score == MAX_SCORE:
            # A winning move is already known, minimax is not necessary.
            # This also prevents possible loops.
            best_score = MAX_SCORE
            
            for child in self.root.children:    
                if child.score == best_score:
                    best_child = child
                    break
        else:
            depth = 1
            while time.time() < end_timestamp:
                curr_best_score, curr_best_child = self.minimax(self.root, depth, -np.inf, +np.inf, end_timestamp)
                if curr_best_score is None or curr_best_child is None:
                    break
                best_score, best_child = curr_best_score, curr_best_child
                if best_child.score != best_score:
                    logging.error("ERROR, returned the wrong child")
                depth += 1
        
        if self.__debug:
            logging.debug(f"Explored depth = {depth}")
            logging.debug(f"Explored nodes: {self.__explored_nodes}, {self.__explored_nodes/(timeout):.2f} nodes/s")

        
        self.root = best_child
        self.state.applyMove(best_child.start, best_child.end)
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
                    # Move found, update the root and
                    # leave the board status as is (do not need to revert).
                    self.root = child
                    return
                self.state.revertMove(child.start, child.end, captured)
        except:
            logging.error("Error in applying opponent move")
        
        # Move not found among the children of the root,
        # the current tree is deleted.
        self.state = next_state
        self.root = TreeNode(None, None, None)


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

            best_child : TreeNode
                Best children's score found.
    """
    def minimax(self, 
        tree_node:TreeNode, 
        max_depth:int, 
        alpha:float, beta:float, 
        timeout_timestamp:float) -> tuple[float|None, TreeNode|None]:
        if self.__debug:
            self.__explored_nodes += 1

        best_child = None

        if self.state.getGameState() != OPEN or max_depth == 0:
            eval = self.state.evaluate(self.player_color)
        else:
            if ((self.state.is_white_turn and self.player_color == WHITE) or
                (not self.state.is_white_turn and self.player_color == BLACK)):
                # Max
                eval = -np.inf
                for child in tree_node.getChildren(self.state):
                    captured = self.state.applyMove(child.start, child.end)
                    
                    if time.time() >= timeout_timestamp:
                        # Times up
                        # Drop the currently generated children 
                        # as the generation is not complete
                        tree_node.children = None
                        return None, None

                    eval_minimax, _ = self.minimax(child, max_depth-1, alpha, beta, timeout_timestamp)
                    if eval_minimax is None: return None, None # Times up
                    
                    eval = max(eval, eval_minimax)
                    if eval > alpha: # Also saves best move for the choice at the root
                        alpha = eval
                        best_child = child

                    self.state.revertMove(child.start, child.end, captured)
                    if eval >= beta: break # cutoff
            else:
                # Min
                eval = np.inf
                for child in tree_node.getChildren(self.state):
                    captured = self.state.applyMove(child.start, child.end)

                    if time.time() >= timeout_timestamp:
                        # Times up
                        tree_node.children = None
                        return None, None

                    eval_minimax, _ = self.minimax(child, max_depth-1, alpha, beta, timeout_timestamp)
                    if eval_minimax is None: return None, None # Times up

                    eval = min(eval, eval_minimax)
                    beta = min(eval, beta)

                    self.state.revertMove(child.start, child.end, captured)
                    if eval <= alpha: break # cutoff

        tree_node.score = eval
        return eval, best_child
        
