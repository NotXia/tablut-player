from .State import State
import cython
import logging
logger = logging.getLogger(__name__)
if not cython.compiled: logger.warn(f"Using non-compiled {__file__} module")


EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


class TraspositionEntry:
    def __init__(self, entry_type:EXACT|LOWERBOUND|UPPERBOUND, value:float):
        self.type = entry_type
        self.value = value

    def __str__(self):
        return f"{'E' if self.type == EXACT else 'L' if self.type == LOWERBOUND else 'U'}, {self.value}"


class TranspositionTable:
    """
        Parameters
        ----------
            max_size : int
                The maximum number of entries the table can hold.
    """
    def __init__(self, max_size:int):
        self.max_size = max_size
        self.table = {} # Dictionary keys follows the insertion order

    def __setitem__(self, state:State, entry:TraspositionEntry):
        board_hash = hash(state)
        
        if board_hash in self.table:
            # Renew the entry
            del self.table[board_hash]
            self.table[board_hash] = entry
        else:
            # Drop the oldest entry if needed
            if len(self.table) >= self.max_size:
                oldest_board = next(iter(self.table.keys()))
                del self.table[oldest_board]
            self.table[board_hash] = entry

    def __getitem__(self, state:State) -> TraspositionEntry|None:
        return self.table.get(hash(state), None)
    
    def __str__(self):
        return str(f"{len(self.table)}/{self.max_size} {[f'({k} | {self.table[k]})' for k in self.table]}")
