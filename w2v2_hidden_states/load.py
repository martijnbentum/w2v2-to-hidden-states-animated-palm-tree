import librosa
import os
import torch
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2Model
from transformers import Wav2Vec2ForPreTraining as pt
from transformers import AutoModel

default_cache_directory='/vol/tensusers/mbentum/hidden_states/cache'
if not os.path.exists(default_cache_directory):
    default_cache_directory = '/Users/martijn.bentum/hidden_states/cache'
if not os.path.exists(default_cache_directory):
    default_cache_directory = '/home/mb/hidden_states/cache'
pretrained_small = 'facebook/wav2vec2-xls-r-300m'
pretrained_big = 'facebook/wav2vec2-xls-r-2b'
default_checkpoint = 'facebook/wav2vec2-xls-r-300m'
hubert_base = 'facebook/hubert-base-ls960'

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
    # model = Wav2Vec2Model.from_pretrained(checkpoint,
    model = AutoModel.from_pretrained(checkpoint,
        cache_dir = cache_directory)
    if gpu: model.to('cuda')
    return model

def load_model_pt(checkpoint = None, gpu = False):
    if not checkpoint: checkpoint = default_checkpoint
    model_pt = pt.from_pretrained(checkpoint)
    if gpu: model_pt.to('cuda')
    return model_pt

def load_hubert_base_model(cache_directory = None, gpu = False):
    return load_pretrained_model(hubert_base, cache_directory, gpu)

