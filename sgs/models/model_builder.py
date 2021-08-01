#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch

from sgs.models.video_model_builder import SlowFastModel, ResNet3D

# Supported model types
_MODEL_TYPES = {
    "slowfast": SlowFastModel,
    '3dresnet': ResNet3D,
}


def build_model(cfg):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in sgs/config/defaults.py.
    """
    assert (
        cfg.MODEL.ARCH in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.ARCH)
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    model = _MODEL_TYPES[cfg.MODEL.ARCH](cfg)
    # Determine the GPU used by the current process

    return model
