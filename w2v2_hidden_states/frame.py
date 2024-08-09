from . import codebook
import copy
import numpy as np

class Frames:
    '''Frames class to handle the frames of the wav2vec2 outputs 
    '''
    def __init__(self,n_frames, stride = 1/49, field = 0.025, 
        start_time = 0, frames = None, identifier = '', name = '',
        audio_filename = '', outputs = None, codebook_indices = None,
        codebook = None):
        '''Handles the frames of the wav2vec2 outputs and codebook indices
        n_frames        number of frames
        stride          the time between frames
        field           the length of the frame
        start_time      the start time of the first frame
        frames          list of frames
        identifier      identifier of the outputs 
        name            name of the outputs
        audio_filename  audio filename
        outputs         the outputs of the wav2vec2 model
        codebook_ind... the codebook indices ie the quantized representation
                        of the cnn output as indices in the codebook
        codebook        the codebook ie the set of quantized representations
                        of the cnn output
        '''
        
        self.identifier = identifier
        self.name = name
        self.audio_filename = audio_filename
        self.outputs = outputs
        self.ci= codebook_indices
        self.codebook = codebook
        self.stride = stride
        self.field = field
        self.n_frames = n_frames
        self.start_time = start_time
        self._make_frames()
        self.end_time = self.frames[-1].end_time 
        self.duration = self.end_time - self.frames[0].start_time

    def __repr__(self):
        m = f'Frames(#frames:{self.n_frames}, start:{self.start_time:.3f}'
        m += f', duration:{self.duration:.3f}'
        m += f', stride:{self.stride:.3f}, field:{self.field})'
        return m

    def _make_frames(self):
        self.frames = []
        for index in range(self.n_frames):
            self.frames.append(Frame(index, self.stride, self.field, 
                self.start_time, self))

    def select_frames(self,start,end, percentage_overlap = None):
        '''Select frames that overlap with the start and end times
        start               start time
        end                 end time
        percentage_overlap  the percentage of the frame that must overlap
                            with the segment between start and end time
        '''
        selected_frames = []
        for frame in self.frames:
            if percentage_overlap == None:
                if frame.overlap(start,end):
                    selected_frames.append(frame)
            elif frame.overlap_percentage(start,end) >= percentage_overlap:
                selected_frames.append(frame)
        return selected_frames

    def cnn(self, start_time, end_time, average = False):
        '''Get the cnn output of the frames that overlap with the start 
        and end times
        '''
        frames = self.select_frames(start_time, end_time)
        output = np.array([frame.cnn() for frame in frames])
        if average:return np.mean(output,axis=0)
        return output
    
    def feature_vector(self, start_time, end_time, average = False):
        '''Get the cnn output of the frames that overlap with the start 
        and end times
        '''
        return self.cnn(start_time, end_time, average)

    def transformer(self, layer, start_time, end_time, average = False):
        '''Get the transformer output of the frames that overlap with the start 
        and end times
        '''
        frames = self.select_frames(start_time, end_time)
        output = np.array([frame.transformer(layer) for frame in frames])
        if average:return np.mean(output,axis=0)
        return output

    def codebook_indices(self, start_time, end_time):
        '''Get the codebook indices of the frames that overlap with the start
        and end times
        codebook indices are indices in the codebook with quantized 
        representation of the cnn output
        '''
        frames = self.select_frames(start_time, end_time)
        ci = np.array([frame.codebook_indices() for frame in frames])
        return ci

    def codevectors(self, start_time, end_time):
        '''Get the codevectors of the frames that overlap with the start
        and end times
        '''
        codebook_indices = self.codebook_indices(start_time, end_time)
        return codebook.multiple_codebook_indices_to_codevectors(
            codebook_indices, self.codebook)


class Frame:
    '''Frame class to handle a single frame of the wav2vec2 output.'''
    def __init__(self, index, stride, field, global_start_time, parent):
        '''Handles a single frame of the wav2vec2 outputs
        index               index of the frame
        stride              the time between frames
        field               the length of the frame
        global_start_time   the start time of the first frame
        parent              the parent Frames object
        '''
        self.index = index
        self.stride = stride
        self.field = field
        self.global_start_time = global_start_time
        self.parent = parent
            
        self.start_time = self.index * self.stride + self.global_start_time
        self.end_time = self.start_time + self.field

    def __repr__(self):
        return f'Frame({self.index}, {self.start_time:.3f}, {self.end_time:.3f})'

    def overlap(self,start,end):
        return self.start_time < end and self.end_time > start

    def overlap_time(self,start,end): 
        return min(self.end_time,end) - max(self.start_time,start)

    def overlap_percentage(self,start,end):
        if self.overlap(start,end):
            ot = self.overlap_time(start,end)
            return round(ot / self.field,5) * 100
        return 0.0

    def cnn(self):
        if not hasattr(self.parent,'outputs'):
            return None
        return self.parent.outputs.extract_features[0,self.index,:]

    def feature_vector(self):
        return self.cnn()

    def transformer(self, layer):
        if not hasattr(self.parent,'outputs'):
            return None
        return self.parent.outputs.hidden_states[layer][0,self.index,:]

    def codebook_indices(self):
        if not hasattr(self.parent,'ci'):
            return None
        return self.parent.ci[self.index]


def extract_info_from_outputs(outputs):
    '''extracts information from the outputs object.'''
    d = {}
    d['audio_filename'] = outputs.audio_filename
    d['start_time'] = outputs.start_time
    d['identifier'] = outputs.identifier
    d['name'] = outputs.name
    d['outputs'] = outputs
    return d

def make_frames(outputs = None, codebook_indices = None, codebook = None,
    start_time = 0):
    '''make frames object from the outputs and or codebook indices.
    outputs         the outputs of the wav2vec2 model
    codebook_ind... the quantized representation of the cnn output as indices
                    in the codebook
    codebook        the set of quantized representations of the cnn output
    start_time      the start time of the first frame, only used if
                    only codebook indices are provided
                    outputs object has start time information
    '''
    if not outputs and not codebook_indices:
        print('No outputs or codebook indices')
        return None
    if outputs:
        output_n_frames = outputs.extract_features.shape[1]
    if codebook_indices:
        codebook_n_frames = len(codebook_indices)
        print(codebook_n_frames)
    if outputs and codebook_indices:
        assert output_n_frames == codebook_n_frames
    n_frames = output_n_frames if outputs else codebook_n_frames
    info = extract_info_from_outputs(outputs) if outputs else {}
    if info == {}: info['start_time'] = start_time
    frames = Frames(n_frames, codebook_indices = codebook_indices,
        codebook = codebook, **info)
    return frames
    
def make_frames_with_outputs(outputs):
    '''make frames object from the outputs object.'''
    return make_frames(outputs = outputs)

def make_frames_with_codebook_indices(codebook_indices, codebook = None,
    start_time = 0):
    '''make frames object from the codebook indices.
    provide start time to be able to extract correct frames
    provide codebook to be able to extract codevector 
    '''
    return make_frames(codebook_indices = codebook_indices, 
        codebook = codebook, start_time = start_time)

def extract_codevector_times(codebook_indices, start_time, end_time,
    codebook, original_start_time = 0):
    '''extract the codevectors that overlap with the start and end times
    '''
    frames = make_frames_with_codebook_indices(codebook_indices, 
        start_time = original_start_time, codebook = codebook)
    codevectors = frames.codevectors(start_time, end_time)
    selected_frames = frames.select_frames(start_time, end_time)
    start_time = selected_frames[0].start_time
    end_time = selected_frames[-1].end_time
    return codevectors, start_time, end_time

def extract_codebook_indices_times(codebook_indices, start_time, 
    end_time, original_start_time = 0):
    '''extract the codebook indices that overlap with the start and end times
    provide original_start_time to be able to extract correct frames
    codebook_indices   the quantized representation of the cnn output as indices
    start_time         start time
    end_time           end time
    original_start...  the start time of the first frame
    '''
    frames = make_frames_with_codebook_indices(codebook_indices, 
        start_time = original_start_time)
    codebook_indices = frames.codebook_indices(start_time, end_time)
    selected_frames = frames.select_frames(start_time, end_time)
    start_time = selected_frames[0].start_time
    end_time = selected_frames[-1].end_time
    return codebook_indices, start_time, end_time

def extract_outputs_times(outputs, start_time, end_time):
    '''extract the outputs that overlap with the start and end times
    '''
    frames = make_frames_with_outputs(outputs)
    selected_frames = frames.select_frames(start_time, end_time)
    start_index = selected_frames[0].index
    end_index = selected_frames[-1].index + 1
    start_time = selected_frames[0].start_time
    return extract_outputs_indices(outputs, start_index, end_index, start_time)
    
def extract_outputs_indices(outputs, start_index, end_index, start_time):
    '''extract the outputs that overlap with the start and end indices
    '''
    o = copy.deepcopy(outputs)
    o.audio_filename = outputs.audio_filename
    o.start_time = start_time
    o.identifier = outputs.identifier
    o.name = outputs.name
    o.extract_features = outputs.extract_features[:,start_index:end_index,:]
    o.hidden_states = extract_hidden_states(outputs.hidden_states,
        start_index, end_index)
    return o

def extract_hidden_states(hidden_states, start_index, end_index):
    '''extract the hidden states that overlap with the start and end indices
    '''
    hs = []
    for hidden_state in hidden_states:
        hs.append(hidden_state[:,start_index:end_index,:])
    return hs

