import socket
import struct
import json
import sys
import numpy as np
from State import BLACK, WHITE, EMPTY, KING, State

def recvall(sock, n):
    # Funzione ausiliaria per ricevere n byte o restituire None se viene raggiunta la fine del file (EOF)
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def main():
    # Crea un socket TCP
    color = sys.argv[1]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        ip_addr = sys.argv[2]
        # Imposta l'indirizzo del server in base al colore del giocatore
        if color == 'white':
            server_address = (ip_addr, 5800)
        elif color == 'black':
            server_address = (ip_addr, 5801)
        else:
            raise Exception("Devi giocare come bianco o nero")
        
        # Connessione al server
        sock.connect(server_address)

        # Invia la lunghezza del nome del giocatore e il nome stesso al server
        player_name = 'TheCatIsOnTheTablut'
        sendToServer(player_name, sock)

        # Ciclo infinito di ricezione stato e invio mossa
        test = True
        while True:
            # Ricevi la lunghezza dei dati di stato corrente dal server e poi i dati stessi
            len_bytes = struct.unpack('>i', recvall(sock, 4))[0]
            current_state_server_bytes = sock.recv(len_bytes)

            # Decodifica i dati di stato in formato JSON
            json_current_state_server = json.loads(current_state_server_bytes)
            board = json_current_state_server['board']
            turn = json_current_state_server['turn']

            # Converti nel nostro stato
            state = []
            for row in board:
                r = []
                for col in row:
                    if col == 'EMPTY':
                        r.append(EMPTY)
                    elif col == 'BLACK':
                        r.append(BLACK)
                    elif col == 'WHITE':
                        r.append(WHITE)
                    elif col == 'KING':
                        r.append(KING)
                    else:
                        raise Exception("Stato non riconosciuto")
                state.append(r)
            s = State(np.array(state, dtype=np.byte), turn == 'WHITE')


            # CALCOLA MOSSA
            # Assunzione mossa ((from), (to))
            if test:
                best_move = ((0, 3), (1, 3))
                test = not test
            else:
                best_move = ((1, 3), (0, 3))
                test = not test

            # Converte la mossa del giocatore nel formato richiesto dal server e la invia
            mossa = {
                "from" : fromIndexToLetters(best_move[0]),
                "to" : fromIndexToLetters(best_move[1]),
                "turn" : "W" if color == "white" else "B"
            }
            sendToServer(json.dumps(mossa), sock)

def sendToServer(data, socket):
    # Invia la lunghezza dei dati di stato corrente dal server e poi i dati stessi
    data_bytes = data.encode()
    socket.send(struct.pack('>i', len(data_bytes)))
    socket.send(data_bytes)

def fromIndexToLetters(position):
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    return letters[position[1]] + str(position[0] + 1)

main()
