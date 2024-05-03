import load
import pickle
import socket
import time


def create_socket(port):
    # Create a socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to the server
    s.connect(('127.0.0.1', port))
    return s

def to_vector(audio_filename, port = 3000):
    start = time.time()
    print('loading audio', time.time() - start)
    audio = load.load_audio(audio_filename)
    print('creating socket',port,time.time() - start)
    socket = create_socket(port)
    print('sending audio', time.time() - start)
    socket.sendall(pickle.dumps(audio))
    print('done sending', time.time() - start)
    return receive_vector(socket, start)

def receive_vector(socket, start = 0):
    print('receiving', time.time() - start)
    rb = b''
    while True:
        data = socket.recv(1024)
        if len(data) < 1024: 
            rb += data
            break
        rb += data
    print('done receiving', time.time() - start)
    socket.close()
    return pickle.loads(rb)
