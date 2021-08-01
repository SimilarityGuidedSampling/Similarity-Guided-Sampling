#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""


import torch

from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from .build import build_dataset


def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            sgs/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )

    return loader


def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
