import librosa
import os
import torch
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2Model

default_cache_directory='/vol/tensusers/mbentum/hidden_states/cache'
if not os.path.exists(default_cache_directory):
    default_cache_directory = '/Users/martijn.bentum/hidden_states/cache'
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

def load_pretrained_model(checkpoint = None, cache_directory = None, 
        gpu = False):
    if not checkpoint: checkpoint = default_checkpoint
    if not cache_directory: cache_directory = default_cache_directory
    model = Wav2Vec2Model.from_pretrained(checkpoint,
        cache_dir = cache_directory)
    if gpu: model.to('cuda')
    return model
