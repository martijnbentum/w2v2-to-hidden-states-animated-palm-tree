import librosa
import locations
import torch
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2Model

default_cache_directory='/vol/tensusers/mbentum/CGN_AUDIO_EXTRACT/WAV2VEC_DATA/'
default_checkpoint = 'facebook/wav2vec2-xls-r-300m'

def load_audio(filename, start = 0.0, end=None):
	if not end: duration = None
	else: duration = end - start
	audio, sr = librosa.load(filename, sr = 16000, offset=start, 
        duration=duration)
	return audio

def load_pretrained_processor_model(checkpoint = None, cache_directory = None):
    if not checkpoint: checkpoint = default_checkpoint
    if not cache_directory: cache_directory = default_cache_directory
    if checkpoint == default_checkpoint:
        processor_checkpoint = '../checkpoint'
    else: processor_checkpoint = checkpoint
    processor = Wav2Vec2Processor.from_pretrained(processor_checkpoint)
    model = Wav2Vec2Model.from_pretrained(checkpoint,
        cache_dir = cache_directory)
    return processor, model

def audio_to_pretrained_outputs(audio_filename, start=None, end=None, 
    processor=None, model=None, frame_duration = None):
    if not processor or not model: 
        print('loading pretrained model: facebook/wav2vec2-xls-r-2b')
        processor, model = load_pretrained_processor_model()
    array = audio.load_audio(audio_filename,start, end)
    inputs = processor(array, sampling_rate=16_000, return_tensors='pt',
        padding= True)
    with torch.no_grad():
        outputs = model(**inputs,output_hidden_states = True)
    return outputs
