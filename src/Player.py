import socket
import struct
import json
import numpy as np
from State import BLACK, WHITE, EMPTY, KING, State
from Tree import Tree
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)


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
def initServerConnection(player_name, ip_addr="localhost", port=None):
    # Crea un socket TCP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Imposta l'indirizzo del server in base al colore del giocatore
    if my_color == WHITE:
        server_address = (ip_addr, port if port is not None else 5800)
    elif my_color == BLACK:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Best Tablut player of the world (when it doesn't lose)")
    parser.add_argument("-i", "--ip", type=str, default="localhost", help="IP address of the hosting server")
    parser.add_argument("-p", "--port", type=str, default=None, help="Port of the hosting server")
    parser.add_argument("-c", "--color", type=str.lower, required=True, choices=["white", "black"], help="Color of the player")
    args = parser.parse_args()

    my_color = WHITE if args.color == "white" else BLACK
    player_name = 'TheCatIsOnTheTablut'
    game_tree = None

    sock = initServerConnection(player_name, args.ip, args.port)


    # Ciclo infinito di ricezione stato e invio mossa
    # TODO handle game end
    while True:
        turn, board = receiveStateFromServer(sock)
        curr_turn = WHITE if turn == "white" else BLACK

        if curr_turn != my_color:
            continue

        # Converti nel nostro stato
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
                    logging.error(f"Unknown board value\n{curr_board}")
                    raise Exception("Stato non riconosciuto")
        curr_state = State(curr_board, curr_turn == WHITE)

        if game_tree is None:
            # Tree created for the first time
            game_tree = Tree(curr_state, my_color)
        else:
            game_tree.applyOpponentMove(curr_state)

        start_pos, end_pos = game_tree.decide(0)
        logging.info(f"Best move {start_pos} -> {end_pos}")

        sendMoveToServer(sock, start_pos, end_pos, my_color)