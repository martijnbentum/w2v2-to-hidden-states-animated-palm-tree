    

class Frames:
    def __init__(self,n_frames, stride = 1/49, field = 0.025, 
        start_time = 0, frames = None, identifier = '', name = ''):
        self.identifier = identifier
        if frames: self._init_with_frames(frames, identifier)
        else:
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

    def _init_with_frames(self,frames, identifier):
        self.frames = frames
        self.n_frames = len(frames)
        self.stride = frames[0].stride
        self.field = frames[0].field
        self.start_time = frames[0].start_time
        self.end_time = frames[-1].end_time 
        self.duration = self.end_time - self.start_time

    def _make_frames(self):
        self.frames = []
        for index in range(self.n_frames):
            self.frames.append(Frame(index, self.stride, self.field, 
                self.start_time))

    def select_frames(self,start,end, percentage_overlap = None):
        selected_frames = []
        for frame in self.frames:
            if percentage_overlap == None:
                if frame.overlap(start,end):
                    selected_frames.append(frame)
            elif frame.overlap_percentage(start,end) >= percentage_overlap:
                selected_frames.append(frame)
        # return selected_frames
        return Frames(n_frames = len(selected_frames), frames = selected_frames)

class Frame:
    def __init__(self, index, stride, field, global_start_time):
        self.index = index
        self.stride = stride
        self.field = field
        self.global_start_time = global_start_time
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

