import requests

# Define the URL of the server
default_url = 'http://127.0.0.1:8080'

'''
# Define the data to be sent in the POST request
data = {'name': 'Alice'}

# Send the POST request
response = requests.post(url, data=data)

# Print the response from the server
print(response.text)
'''


def send_dict_to_server(d, url = default_url):
    # Send the filenames to the server
    response = requests.post(url, data=d)
    return response


def send_audio_output_filenames_to_server(audio_filename,
    output_filename, url = default_url):
    d = {'audio_filename': audio_filename, 'output_filename': output_filename}
    response = send_dict_to_server(d, url)
    return response
