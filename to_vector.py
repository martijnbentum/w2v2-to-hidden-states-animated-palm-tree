import load
import torch

def audio_to_vector(audio, model=None, feature_extractor = None,
    gpu = False):
    if not model: 
        model = load_pretrained_model(gpu = gpu)
    if not feature_extractor:
        feature_extractor = load.make_feature_extractor()
    array = audio
    inputs = feature_extractor(array, sampling_rate=16_000, return_tensors='pt',
        padding= True)
    if gpu:inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs,output_hidden_states = True)
    return outputs

def filename_to_vector(audio_filename, start=None, end=None, 
    model=None, feature_extractor = None, gpu = False):
    array = load.load_audio(audio_filename,start, end)
    return audio_to_vector(array, model, feature_extractor, gpu)
