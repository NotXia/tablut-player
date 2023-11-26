DEF DEBUG = True

cimport cython
from cpython cimport array
import numpy as np
cimport numpy as cnp
cnp.import_array()
from .State cimport State, OPEN, WHITE, BLACK, MAX_SCORE, MIN_SCORE
from .TreeNode cimport TreeNode
from libc.time cimport time, time_t
import logging
logger = logging.getLogger(__name__)

cdef score_t TIMEOUT = MIN_SCORE - 1000.0
cdef score_t PLUS_INFINITY = MAX_SCORE + 100.0
cdef score_t MINUS_INFINITY = MIN_SCORE - 100.0


"""
    Class that represents the whole game tree.
"""
cdef class Tree():
    def __init__(self, State initial_state, char player_color, dict weights, debug=False):
        self.state = initial_state
        self.player_color = player_color
        self.root = TreeNode(NULL_COORD, NULL_COORD)
        self.turns_count = 0

        self.early_positive_weights = array.array("f", weights["early"]["positive"])
        self.early_negative_weights = array.array("f", weights["early"]["negative"])
        self.mid_positive_weights = array.array("f", weights["mid"]["positive"])
        self.mid_negative_weights = array.array("f", weights["mid"]["negative"])
        self.late_positive_weights = array.array("f", weights["late"]["positive"])
        self.late_negative_weights = array.array("f", weights["late"]["negative"])
        self.curr_positive_weights = self.early_positive_weights
        self.curr_negative_weights = self.early_negative_weights

        IF DEBUG:
            self.__explored_nodes = 0


    """
        Determines the next best move.

        Parameters
        ----------
            timeout : score_t
                Seconds available to make a choice.

        Returns
        -------
            from : tuple[int, int]
                Starting coordinates of the move.

            to : tuple[int, int]
                Ending coordinates of the move.
                
            best_score : score_t
                Score of the chosen move.
    """
    cpdef tuple[Coord, Coord, score_t] decide(self, int timeout):
        IF DEBUG: 
            self.__explored_nodes = 0
        
        self.turns_count += 1
        cdef time_t end_timestamp = time(NULL) + timeout
        cdef TreeNode best_child = None, child
        cdef int depth = 0
        cdef score_t best_score = MIN_SCORE, curr_best_score

        self.__updateWeights()
        
        if self.root.score == MAX_SCORE:
            # A winning move is already known, minimax is not necessary.
            # This also prevents possible loops.
            IF DEBUG: 
                logger.debug("Following winning path")
            best_score = MAX_SCORE
            
            for child in self.root.children:    
                if child.score == best_score:
                    best_child = child
                    break
        else:
            while time(NULL) < end_timestamp:
                depth += 1
                curr_best_score = self.minimax(self.root, depth, MINUS_INFINITY, PLUS_INFINITY, end_timestamp)
                if curr_best_score == TIMEOUT:
                    depth -= 1
                    break
                best_score = curr_best_score

                for child in self.root.children:
                    if child.score == best_score:
                        best_child = child
                        break
        
        IF DEBUG: 
            logger.debug(f"Explored depth = {depth}")
            logger.debug(f"Explored nodes: {self.__explored_nodes}, {self.__explored_nodes/(timeout):.2f} nodes/s")
        
        self.root = best_child
        _ = self.state.applyMove(best_child.start, best_child.end)
        return best_child.start, best_child.end, best_score
    

    """
        Updates the weights used to compute the heuristics.
    """
    cdef void __updateWeights(self):
        if self.turns_count <= 5:
            self.curr_positive_weights = self.early_positive_weights
            self.curr_negative_weights = self.early_negative_weights
        elif self.turns_count <= 15:
            self.curr_positive_weights = self.mid_positive_weights
            self.curr_negative_weights = self.mid_negative_weights
        else:
            self.curr_positive_weights = self.late_positive_weights
            self.curr_negative_weights = self.late_negative_weights


    """
        Moves the root of the tree to the node containing the opponent's move.
        If it does not exist, the tree is resetted.

        Parameters
        ----------
            next_state : State
                State of the board after the opponent's move.
    """
    def applyOpponentMove(self, next_state: State):
        cdef TreeNode child
        cdef list[tuple[captured, char]] captured

        try:
            for child in self.root.children:
                captured = self.state.applyMove(child.start, child.end)
                if np.all(self.state.board == next_state.board):
                    IF DEBUG:
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
        IF DEBUG:
            logger.debug("Dropping tree")
        self.state = next_state
        self.root = TreeNode(NULL_COORD, NULL_COORD)



    """
        Runs minimax with alpha-beta pruning on a given node.

        Parameters
        ----------
            tree_node : TreeNode
                Node to explore.

            max_depth : int
                Maximum reachable depth before evaluating the node.

            alpha, beta : score_t
                Alpha and beta for pruning

        Returns
        -------
            best_score : score_t
                Best children's score found.
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef score_t minimax(self, TreeNode tree_node, int max_depth, score_t alpha, score_t beta, time_t timeout_timestamp):
        if time(NULL) >= timeout_timestamp: return TIMEOUT # Timeout
        IF DEBUG:
            self.__explored_nodes += 1

        cdef TreeNode child
        cdef score_t eval_minimax, eval
        
        if self.state.getGameState() != OPEN or max_depth == 0:
            eval = self.state.evaluate(self.player_color, self.curr_positive_weights, self.curr_negative_weights)
        else:
            if ((self.state.is_white_turn and self.player_color == WHITE) or
                (not self.state.is_white_turn and self.player_color == BLACK)):
                # Max
                eval = MINUS_INFINITY
                for child in tree_node.getChildren(self.state):
                    captured = self.state.applyMove(child.start, child.end)
                    eval_minimax = self.minimax(child, max_depth-1, alpha, beta, timeout_timestamp)
                    self.state.revertMove(child.start, child.end, captured)
                    
                    if eval_minimax == TIMEOUT: return TIMEOUT # Timeout
                    
                    eval = max(eval, eval_minimax)
                    alpha = max(eval, alpha)
                    if eval >= beta: break # cutoff
            else:
                # Min
                eval = PLUS_INFINITY
                for child in tree_node.getChildren(self.state):
                    captured = self.state.applyMove(child.start, child.end)
                    eval_minimax = self.minimax(child, max_depth-1, alpha, beta, timeout_timestamp)
                    self.state.revertMove(child.start, child.end, captured)
                    
                    if eval_minimax == TIMEOUT: return TIMEOUT # Timeout

                    eval = min(eval, eval_minimax)
                    beta = min(eval, beta)
                    if eval <= alpha: break # cutoff

        tree_node.score = eval
        return eval
        
