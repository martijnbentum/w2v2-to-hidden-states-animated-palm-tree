import load
import torch
from pathlib import Path

def audio_to_vector(audio, model=None, feature_extractor = None,
    gpu = False, numpify_output = True):
    if not model: 
        model = load.load_pretrained_model(gpu = gpu)
    if not feature_extractor:
        feature_extractor = load.make_feature_extractor()
    array = audio
    inputs = feature_extractor(array, sampling_rate=16_000, return_tensors='pt',
        padding= True)
    if gpu:inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs,output_hidden_states = True)
    return numpify(outputs)

def filename_to_vector(audio_filename, start=None, end=None, 
    model=None, feature_extractor = None, gpu = False, 
        identifier = '', name = '', numpify_output = True):
    audio_filename = Path(audio_filename).resolve()
    array = load.load_audio(audio_filename,start, end)
    outputs = audio_to_vector(array, model, feature_extractor, gpu,
        numpify_output)
    outputs = add_info(outputs, audio_filename, start, identifier, name)
    return outputs


def add_info(outputs, audio_filename, start, identifier, name):
    if start == None: start = 0
    audio_filename = str(audio_filename)
    outputs.audio_filename = audio_filename
    outputs.start_time = start
    outputs.identifier = identifier
    outputs.name = name
    return outputs

def numpify(outputs):
    outputs.extract_features = outputs.extract_features.cpu().numpy()
    hs = []
    for hidden_state in outputs.hidden_states:
        hs.append(hidden_state.cpu().numpy())
    outputs.hidden_states = hs
    return outputs
