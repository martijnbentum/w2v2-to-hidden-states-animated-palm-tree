from . import load
import torch
from pathlib import Path
from transformers.modeling_outputs import BaseModelOutput

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
    if not hasattr(outputs, 'extract_features'):
        if 'Hubert' in str(type(model)):
            o = audio_to_features(audio, model, feature_extractor, gpu)
            outputs.extract_features = o
    return numpify(outputs)

def filename_to_vector(audio_filename, start=0.0, end=None, 
    model=None, feature_extractor = None, gpu = False, 
        identifier = '', name = '', numpify_output = True):
    audio_filename = Path(audio_filename).resolve()
    array = load.load_audio(audio_filename,start, end)
    outputs = audio_to_vector(array, model, feature_extractor, gpu,
        numpify_output)
    outputs = add_info(outputs, audio_filename, start, end, identifier, name)
    return outputs


def add_info(outputs, audio_filename, start, end, identifier, name):
    audio_filename = str(audio_filename)
    outputs.audio_filename = audio_filename
    outputs.start_time = start
    outputs.end_time = end
    outputs.identifier = identifier
    outputs.name = name
    return outputs

def numpify(outputs):
    if hasattr(outputs, 'extract_features'):
        if type(outputs.extract_features) == torch.Tensor:
            outputs.extract_features = outputs.extract_features.cpu().numpy()
    hs = []
    for hidden_state in outputs.hidden_states:
        hs.append(hidden_state.cpu().numpy())
    outputs.hidden_states = hs
    return outputs

def audio_to_features(audio, model=None, feature_extractor = None,
    gpu = False, identifier = '', name = ''):
    if not model: 
        model = load.load_pretrained_model(gpu = gpu)
    if not feature_extractor:
        feature_extractor = load.make_feature_extractor()
    array = audio
    inputs = feature_extractor(array, sampling_rate=16_000, return_tensors='pt',
        padding= True)
    if gpu:inputs = inputs.to('cuda')
    with torch.no_grad():
        input_values = inputs['input_values']
        outputs = model.feature_extractor(input_values)
    outputs = outputs.transpose(1,2).detach().cpu().numpy()
    return outputs

def filename_to_features(audio_filename, start=0.0, end=None, 
    model=None, feature_extractor = None, gpu = False, 
        identifier = '', name = '', numpify_output = True):
    audio_filename = Path(audio_filename).resolve()
    array = load.load_audio(audio_filename,start, end)
    outputs = audio_to_features(array, model, feature_extractor, gpu, 
        identifier, name)
    o = BaseModelOutput(hidden_states = None)
    o.extract_features = outputs
    outputs = add_info(o, audio_filename, start, end, identifier, name)
    return outputs
