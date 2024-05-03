import load
import socket
import to_vector
import pickle

def create_socket(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', port))
    s.listen(1)
    return s

def create_sockets(start = 3000, n = 10):
    return [create_socket(start + i) for i in range(n)]

class Server:
    def __init__(self,port = 3000, model = None, feature_extractor = None, 
        gpu = False):
        self.port = port
        self.model = model
        if not self.model:
            self.model = load.load_pretrained_model(gpu = gpu)
        self.feature_extractor = feature_extractor
        if not self.feature_extractor:
            fe = load.make_feature_extractor()
            self.feature_extractor = fe
        self.gpu = gpu
        self.socket = create_socket(self.port)

    def connect(self):
        print('connecting...')
        self.connection, _ = self.socket.accept()
        self.get_audio()

    def get_audio(self):
        print('receiving audio...')
        audio = b''
        while True:
            data = self.connection.recv(1024)
            if len(data) < 1024: 
                audio += data
                break
            audio += data
        self.audio = pickle.loads(audio)
        self.to_vector()

    def to_vector(self):
        print('converting to vector...')
        outputs = to_vector.audio_to_vector(self.audio, self.model, 
            self.feature_extractor, self.gpu)
        self.send_outputs(outputs)
        

    def send_outputs(self, outputs):
        print('sending outputs...')
        self.connection.sendall(pickle.dumps(outputs))
        self.connection.close()
        print('done')
        self.connect()


    
