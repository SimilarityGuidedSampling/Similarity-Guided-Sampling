import random
import math

import torch

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame_indices):
        for i, t in enumerate(self.transforms):
            if isinstance(frame_indices[0], list):
                next_transforms = Compose(self.transforms[i:])
                dst_frame_indices = [
                    next_transforms(clip_frame_indices)
                    for clip_frame_indices in frame_indices
                ]

                return dst_frame_indices
            else:
                frame_indices = t(frame_indices)
        return frame_indices


def temporal_slicing(frame_indices, clip_size, clip_idx, num_clips):
    """
       Sample a clip of size clip_size from a video of size video_size and
       return the indices of the first and last frame of the clip. If clip_idx is
       -1, the clip is randomly sampled, otherwise uniformly split the video to
       num_clips clips, and select the start and end index of clip_idx-th video
       clip.
       Args:
           frame_indices (int): all frame indices
           clip_size (int): size of the clip to sample from the frames.
           clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
               clip_idx is larger than -1, uniformly split the video to num_clips
               clips, and select the start and end index of the clip_idx-th video
               clip.
           num_clips (int): overall number of clips to uniformly sample from the
               given video for testing.

       """
    delta = max(len(frame_indices) - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    index = torch.linspace(start_idx, end_idx, clip_size)
    index = torch.clamp(index, 0, len(frame_indices) - 1).long()
    if not isinstance(frame_indices, torch.Tensor):
        frame_indices = torch.tensor(frame_indices).long()

    frames = torch.index_select(frame_indices, 0, index)

    return frames


class LoopPadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = list(frame_indices)

        for index in out:
            if len(out) >= self.size:
                return torch.tensor(out)
            out.append(index)

        return torch.tensor(out)


class TemporalBeginCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object):

    def __init__(self, size):
        self.size = size
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        if len(out) < self.size:
            out = self.loop(out)

        return out


class TemporalEvenCrop(object):

    def __init__(self, size, n_samples=1):
        self.size = size
        self.n_samples = n_samples
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        n_frames = len(frame_indices)
        stride = max(
            1, math.ceil((n_frames - 1 - self.size) / (self.n_samples - 1)))

        out = []
        for begin_index in frame_indices[::stride]:
            if len(out) >= self.n_samples:
                break
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)

        return out


class SlidingWindow(object):

    def __init__(self, size, stride=0):
        self.size = size
        if stride == 0:
            self.stride = self.size
        else:
            self.stride = stride
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        out = []
        for begin_index in frame_indices[::self.stride]:
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)

        return out


class TemporalSubsampling(object):

    def __init__(self, stride):
        self.stride = stride

    def __call__(self, frame_indices):
        return torch.tensor(frame_indices[::self.stride])


class Shuffle(object):

    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, frame_indices):
        frame_indices = [
            frame_indices[i:(i + self.block_size)]
            for i in range(0, len(frame_indices), self.block_size)
        ]
        random.shuffle(frame_indices)
        frame_indices = [t for block in frame_indices for t in block]
        return frame_indices