import socket
import struct
import json

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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Imposta l'indirizzo del server in base al colore del giocatore
        if color == 'white':
            server_address = ('localhost', 5800)
        elif color == 'black':
            server_address = ('localhost', 5801)
        else:
            raise Exception("Devi giocare come bianco o nero")
        
        # Connessione al server
        sock.connect(server_address)

        # Invia la lunghezza del nome del giocatore e il nome stesso al server
        player_name_bytes = player_name.encode()
        sock.send(struct.pack('>i', len(player_name_bytes)))
        sock.send(player_name_bytes)

        # Ricevi la lunghezza dei dati di stato corrente dal server e poi i dati stessi
        len_bytes = struct.unpack('>i', recvall(sock, 4))[0]
        current_state_server_bytes = sock.recv(len_bytes)

        # Decodifica i dati di stato in formato JSON
        json_current_state_server = json.loads(current_state_server_bytes)

        # Converte la mossa del giocatore nel formato richiesto dal server e la invia
        move_for_server = convert_move_for_server(move, color)
        move_for_server_bytes = move_for_server.encode()
        sock.send(struct.pack('>i', len(move_for_server_bytes)))
        sock.send(move_for_server_bytes)
