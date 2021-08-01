#!/usr/bin/env python3

import logging
import numpy as np
import time
import torch

import cv2

from . import transform as transform

logger = logging.getLogger(__name__)


def compute_frames_distances(frames, frame_pairs, num_frames, dist_func):
    """
    This function computes the distance between all pairs of frame
    and store the result in matrix

    Args:
        frames:
        dist_func:
        frame_pairs:
        num_frames:

    Returns:
        frames_mse (np.array): matrix of distances between each pair
    """
    # if torch.cuda.is_available():
    #     frames = frames.cuda()
    frames_mse = np.zeros((num_frames, num_frames))
    for pairs in frame_pairs:
        t1, t2 = pairs
        f_t1, f_t2 = torch.chunk(frames[:,[t1,t2], :, :], chunks=2, dim=1)  # CxTxWxH
        mse_val = dist_func(f_t1, f_t2)
        frames_mse[t1, t2] = mse_val.item()
        frames_mse[t2, t1] = mse_val.item()

    return frames_mse



def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.

    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.

    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = [cv2.imread(image_path) for image_path in image_paths]

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))


def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.

    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames

    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq


def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        frames, _ = transform.random_short_side_scale_jitter(
            images=frames,
            min_size=min_scale,
            max_size=max_scale,
            inverse_uniform_sampling=inverse_uniform_sampling,
        )
        frames, _ = transform.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _ = transform.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
    return frames