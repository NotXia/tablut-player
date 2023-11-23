import subprocess

WHITE_WIN = "white win"
BLACK_WIN = "black win"
DRAW = "draw"


class Environment:
    def __init__(self, server_path:str, gui:bool=False):
        self.server_path = server_path
        self.gui = gui

    def startGame(self):
        subprocess.run(["ant", "compile"], cwd = self.server_path, capture_output = True)
        result = subprocess.run(
            ["ant", "gui-server" if self.gui else "server"], 
            cwd = self.server_path,
            capture_output = True, 
            text = True
        )

        white_moves = result.stdout.count("Waiting for W")
        black_moves = result.stdout.count("Waiting for B")

        if "WHITE WIN" in result.stdout:
            winner = WHITE_WIN
        elif "BLACK WIN" in result.stdout:
            winner = BLACK_WIN
        elif "DRAW" in result.stdout:
            winner = DRAW
        else:
            winner = None

        return winner, white_moves, black_moves