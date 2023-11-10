from State import State, OPEN, WHITE, BLACK, MAX_SCORE, MIN_SCORE
import numpy as np
from TreeNode import TreeNode
import logging


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
                Timestamp of when the execution must end.

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
        # TODO Handle timeout
        if self.__debug: 
            self.__explored_nodes = 0

        if self.root.score == MAX_SCORE:
            # A winning move is already known, minimax is not necessary.
            # This also prevents possible loops.
            best_score = MAX_SCORE
        else:
            best_score = self.minimax(self.root, 3, -np.inf, +np.inf)
        
        if self.__debug:
            logging.debug(f"Explored nodes: {self.__explored_nodes}")

        for child in self.root.children:
            if child.score == best_score:
                self.root = child
                self.state.applyMove(child.start, child.end)
                return child.start, child.end, best_score
        

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
    """
    def minimax(self, tree_node:TreeNode, max_depth:int, alpha:float, beta:float):
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
                    captured = self.state.applyMove(child.start, child.end)
                    eval = max(eval, self.minimax(child, max_depth-1, alpha, beta))
                    self.state.revertMove(child.start, child.end, captured)
                    alpha = max(eval, alpha)
                    if eval >= beta: break # cutoff
            else:
                # Min
                eval = np.inf
                for child in tree_node.getChildren(self.state):
                    captured = self.state.applyMove(child.start, child.end)
                    eval = min(eval, self.minimax(child, max_depth-1, alpha, beta))
                    self.state.revertMove(child.start, child.end, captured)
                    beta = min(eval, beta)
                    if eval <= alpha: break # cutoff

        tree_node.score = eval
        return eval
        
