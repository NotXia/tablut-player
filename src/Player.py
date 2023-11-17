import socket
import struct
import json
import numpy as np
from gametree.State import BLACK, WHITE, EMPTY, KING, State
from gametree.Tree import Tree
import time
import logging
logger = logging.getLogger(__name__)


def recvall(sock, n):
    # Funzione ausiliaria per ricevere n byte o restituire None se viene raggiunta la fine del file (EOF)
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def sendToServer(data, socket):
    # Invia la lunghezza dei dati di stato corrente dal server e poi i dati stessi
    data_bytes = data.encode()
    socket.send(struct.pack('>i', len(data_bytes)))
    socket.send(data_bytes)

"""
    Converts our coordinate format into the server's format.
"""
def fromIndexToLetters(position):
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    return letters[position[1]] + str(position[0] + 1)

"""
    Opens the connection to the server and
    does the initial setup.
"""
def initServerConnection(player_name, player_color, ip_addr="localhost", port=None):
    # Crea un socket TCP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Imposta l'indirizzo del server in base al colore del giocatore
    if player_color == WHITE:
        server_address = (ip_addr, port if port is not None else 5800)
    elif player_color == BLACK:
        server_address = (ip_addr, port if port is not None else 5801)
    
    # Connessione al server
    sock.connect(server_address)

    # Invia la lunghezza del nome del giocatore e il nome stesso al server
    sendToServer(player_name, sock)

    return sock

"""
    Waits for the server to send a new board state.
"""
def receiveStateFromServer(sock):
    # Ricevi la lunghezza dei dati di stato corrente dal server e poi i dati stessi
    len_bytes = struct.unpack('>i', recvall(sock, 4))[0]
    current_state_server_bytes = sock.recv(len_bytes)

    # Decodifica i dati di stato in formato JSON
    json_current_state_server = json.loads(current_state_server_bytes)
    board = json_current_state_server['board']
    turn = json_current_state_server['turn'].lower()

    return turn, board

"""
    Sends a move to the server.
"""
def sendMoveToServer(sock, start_pos, end_pos, my_color):
    # Converte la mossa del giocatore nel formato richiesto dal server e la invia
    mossa = {
        "from" : fromIndexToLetters(start_pos),
        "to" : fromIndexToLetters(end_pos),
        "turn" : "W" if my_color == WHITE else "B"
    }
    sendToServer(json.dumps(mossa), sock)

"""
    Parses the board received from the server into our format.
"""
def parseServerBoard(board):
    curr_board = np.zeros((len(board), len(board[0])), dtype=np.byte)

    for i in range(len(board)):
        for j in range(len(board[i])):
            col = board[i][j].upper()
            if col == 'EMPTY' or col == "THRONE":
                curr_board[i, j] = EMPTY
            elif col == 'BLACK':
                curr_board[i, j] = BLACK
            elif col == 'WHITE':
                curr_board[i, j] = WHITE
            elif col == 'KING':
                curr_board[i, j] = KING
            else:
                logger.error(f"Unknown board value\n{curr_board}")
                raise Exception("Stato non riconosciuto")
            
    return curr_board


class Player:
    def __init__(self, 
        my_color: WHITE|BLACK,
        timeout = 60,
        timeout_tol = 1,
        name = "TheCatIsOnTheTablut",
        server_ip = "localhost",
        server_port = None,
        debug = False
    ):
        self.my_color = my_color
        self.name = name
        self.sock = initServerConnection(name, my_color, server_ip, server_port)
        self.timeout = timeout
        self.timeout_tol = timeout_tol
        self.debug = debug

        self.game_tree = None


    def play(self):
        while True:
            turn, board = receiveStateFromServer(self.sock)
            if turn == "white": curr_turn = WHITE
            elif turn == "black": curr_turn = BLACK
            else: break

            if curr_turn != self.my_color:
                continue

            curr_board = parseServerBoard(board)
            curr_state = State(curr_board, curr_turn == WHITE)

            if self.game_tree is None:
                # Tree created for the first time
                self.game_tree = Tree(curr_state, self.my_color, debug=self.debug)
            else:
                self.game_tree.applyOpponentMove(curr_state)

            if self.debug:
                start_time = time.time()
            start_pos, end_pos, score = self.game_tree.decide(self.timeout-self.timeout_tol)
            if self.debug:
                end_time = time.time()
                logger.debug(f"[{end_time-start_time:.2f} s] Best move {start_pos} -> {end_pos} ({score:.3f})")

            sendMoveToServer(self.sock, start_pos, end_pos, self.my_color)

        if turn == "draw":
            print("🇨🇭")
        elif ((turn == "whitewin" and self.my_color == WHITE) or
            (turn == "blackwin" and self.my_color == BLACK)):
            print("ᕦ(⌐■_■)ᕤ")
        else:
            print("(• ᴖ •)")