import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2Model

default_cache_directory='/vol/tensusers/mbentum/hidden_states/cache'
default_checkpoint = 'facebook/wav2vec2-xls-r-300m'

def load_audio(filename, start = 0.0, end=None):
	if not end: duration = None
	else: duration = end - start
	audio, sr = librosa.load(filename, sr = 16000, offset=start, 
        duration=duration)
	return audio

def make_feature_extractor():
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, 
    sampling_rate=16000, padding_value=0.0, do_normalize=True, 
    return_attention_mask=True)
    return feature_extractor

def load_pretrained_model(checkpoint = None, cache_directory = None):
    if not checkpoint: checkpoint = default_checkpoint
    if not cache_directory: cache_directory = default_cache_directory
    model = Wav2Vec2Model.from_pretrained(checkpoint,
        cache_dir = cache_directory)
    return model

def audio_to_pretrained_outputs(audio_filename, start=None, end=None, 
    feature_extractor = None, model=None, frame_duration = None):
    if not model: 
        model = load_pretrained_model()
    if not feature_extractor:
        feature_extractor = make_feature_extractor()
    array = load_audio(audio_filename,start, end)
    inputs = feature_extractor(array, sampling_rate=16_000, return_tensors='pt',
        padding= True)
    with torch.no_grad():
        outputs = model(**inputs,output_hidden_states = True)
    return outputs
