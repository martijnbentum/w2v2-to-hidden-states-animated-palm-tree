import copy
import numpy as np

class Frames:
    def __init__(self,n_frames, stride = 1/49, field = 0.025, 
        start_time = 0, frames = None, identifier = '', name = '',
        audio_filename = '', outputs = None):
        self.identifier = identifier
        self.name = name
        self.audio_filename = audio_filename
        self.outputs = outputs
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
        selected_frames = []
        for frame in self.frames:
            if percentage_overlap == None:
                if frame.overlap(start,end):
                    selected_frames.append(frame)
            elif frame.overlap_percentage(start,end) >= percentage_overlap:
                selected_frames.append(frame)
        return selected_frames

    def cnn(self, start_time, end_time, average = False):
        frames = self.select_frames(start_time, end_time)
        output = np.array([frame.cnn() for frame in frames])
        if average:return np.mean(output,axis=0)
        return output
    
    def feature_vector(self, start_time, end_time, average = False):
        return self.cnn(start_time, end_time, average)

    def transformer(self, layer, start_time, end_time, average = False):
        frames = self.select_frames(start_time, end_time)
        output = np.array([frame.transformer(layer) for frame in frames])
        if average:return np.mean(output,axis=0)
        return output

        

class Frame:
    def __init__(self, index, stride, field, global_start_time, parent):
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


def extract_info_from_outputs(outputs):
    d = {}
    d['audio_filename'] = outputs.audio_filename
    d['start_time'] = outputs.start_time
    d['identifier'] = outputs.identifier
    d['name'] = outputs.name
    d['outputs'] = outputs
    return d
    
def make_frames_with_outputs(outputs):
    n_frames = outputs.extract_features.shape[1]
    info = extract_info_from_outputs(outputs)
    frames = Frames(n_frames, **info)
    return frames

def extract_outputs_times(outputs, start_time, end_time):
    frames = make_frames_with_outputs(outputs)
    selected_frames = frames.select_frames(start_time, end_time)
    start_index = selected_frames[0].index
    end_index = selected_frames[-1].index + 1
    start_time = selected_frames[0].start_time
    return extract_outputs_indices(outputs, start_index, end_index, start_time)
    
def extract_outputs_indices(outputs, start_index, end_index, start_time):
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
    hs = []
    for hidden_state in hidden_states:
        hs.append(hidden_state[:,start_index:end_index,:])
    return hs

