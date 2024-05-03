from http.server import BaseHTTPRequestHandler, HTTPServer
import load
from pathlib import Path
import pickle
import time
import to_vector
from urllib.parse import parse_qs

# Define the HTTP request handler class
class MyHTTPRequestHandler(BaseHTTPRequestHandler):
    '''
    def __init__(self, request, client_address, server, model = None):
        if not model:
            model = load.load_pretrained_model()
        self.model = model
        self.feature_extractor = load.make_feature_extractor()
        super().__init__(request, client_address, server)
    '''
    
    # Handler for GET requests
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<html><body><h1>Hello, world!</h1></body></html>")
        return
    
    # Handler for POST requests
    def do_POST(self):
        # Send response status code
        self.send_response(200)
        # Send headers
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        # Get the length of the content and read it
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        # Parse the POST data
        post_params = parse_qs(post_data.decode('utf-8'))
        self.handle_post_data(post_params)

    def handle_post_data(self, post_params):
        # Extract and print the value of the 'name' parameter
        print(post_params)
        audio_filename, af_status = check_audio_filename_exists(post_params)
        if af_status != 'ok':
            self.wfile.write(f'{audio_filename} {af_status} @'.encode('utf-8'))
            return
        output_filename, of_status = check_output_filename_exists(post_params)
        if of_status == 'file exists':  
            if not check_overwrite(post_params):
                of_status = 'file exists and overwrite is not set'
            else: of_status = 'ok'
        if of_status != 'ok':
            self.wfile.write(f'{audio_filename} {af_status} | '.encode('utf-8'))
            self.wfile.write(f'{output_filename} {of_status} @ '.encode('utf-8'))
            return
        self.wfile.write(f'{audio_filename} {status} |'.encode('utf-8'))
        handle_to_vector(audio_filename, output_filename)
        self.wfile.write(f'{output_filename} ok @'.encode('utf-8'))
        return 

def check_audio_filename_exists(post_params):
    d = {}
    filename = post_params.get('audio_filename', [''])[0]
    if Path(filename).exists(): status = 'ok'
    else: status = 'file not found'
    print(filename,status)
    return filename, status

def check_output_filename_exists(post_params):
    filename = post_params.get('output_filename', [''])[0]
    if not Path(filename).parent.exists():
        status = 'directory does not exist'
    elif Path(filename).exists():
        status = 'file exists'
    else:
        status = 'ok'
    print(filename, status)
    return filename, status

def check_overwrite(post_params):
    overwrite = post_params.get('output_filename', [''])[0]
    print(overwrite,'overwrite')
    return overwrite.lower() == 'true'


def handle_to_vector(self, audio_filename, output_filename):
    o = to_vector.filename_to_vector(audio_filename, model = self.model, 
        feature_extractor = self.feature_extractor )
    with open(output_filename, 'wb') as f:
        pickle.dump(o, f)
 
# Define the host and port to listen on
def make_server(model, host ='127.0.0.1', port = 8080):
    # Create an HTTP server instance
    server = HTTPServer((host, port), MyHTTPRequestHandler)
    server.RequestHandlerClass.model = model
    server.RequestHandlerClass.feature_extractor = load.make_feature_extractor()
    return server

def start_server(server = None, host ='127.0.0.1', port = 8080):
    # Start the server
    if not server:
        server = make_server(host, port)
    print(f'Starting server on {host}:{port}')
    server.serve_forever()
