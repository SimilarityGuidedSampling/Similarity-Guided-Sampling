#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions that handle saving and loading of checkpoints."""

import os
import collections
import pickle
from collections import OrderedDict
import torch

import sgs.utils.logging as logging
from sgs.utils.c2_model_loading import get_name_convert_func

logger = logging.get_logger(__name__)


def make_checkpoint_dir(path_to_job, expr_num):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = os.path.join(path_to_job, "checkpoints", str(expr_num))
    # Create the checkpoint dir from the master process
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir


def get_checkpoint_dir(path_to_job, resume_expr_num):
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "checkpoints", str(resume_expr_num))


def get_path_to_checkpoint(path_to_job, epoch, expr_num):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_job, expr_num), name)


def get_last_checkpoint(path_to_job, resume_expr_num):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    assert resume_expr_num != None, "No experiment number is given"
    d = get_checkpoint_dir(path_to_job, str(resume_expr_num))
    names = os.listdir(d) if os.path.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)


def has_checkpoint(path_to_job, expr_num):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job, str(expr_num))
    files = os.listdir(d) if os.path.exists(d) else []
    return any("checkpoint" in f for f in files)


def is_checkpoint_epoch(cur_epoch, checkpoint_period):
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cur_epoch (int): current number of epoch of the model.
        checkpoint_period (int): the frequency of checkpointing.
    """
    return (cur_epoch + 1) % checkpoint_period == 0


def save_checkpoint(path_to_job, expr_num, model, optimizer, epoch, cfg):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
    """

    # Ensure that the checkpoint dir exists.
    os.makedirs(get_checkpoint_dir(path_to_job, expr_num), exist_ok=True)
    saving_model = model.module if cfg.NUM_GPUS > 1 else model
    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": saving_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint.
    path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1, expr_num)
    torch.save(checkpoint, path_to_checkpoint)
    return path_to_checkpoint


def inflate_weight(state_dict_2d, state_dict_3d):
    """
    Inflate 2D model weights in state_dict_2d to the 3D model weights in
    state_dict_3d. The details can be found in:
    Joao Carreira, and Andrew Zisserman.
    "Quo vadis, action recognition? a new model and the kinetics dataset."
    Args:
        state_dict_2d (OrderedDict): a dict of parameters from a 2D model.
        state_dict_3d (OrderedDict): a dict of parameters from a 3D model.
    Returns:
        state_dict_inflated (OrderedDict): a dict of inflated parameters.
    """
    state_dict_inflated = OrderedDict()
    for k, v2d in state_dict_2d.items():
        assert k in state_dict_3d.keys()
        v3d = state_dict_3d[k]
        # Inflate the weight of 2D conv to 3D conv.
        if len(v2d.shape) == 4 and len(v3d.shape) == 5:
            logger.info("Inflate {}: {} -> {}: {}".format(k, v2d.shape, k, v3d.shape))
            # Dimension need to be match.
            assert v2d.shape[-2:] == v3d.shape[-2:]
            assert v2d.shape[:2] == v3d.shape[:2]
            v3d = v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
        if v2d.shape == v3d.shape:
            v3d = v2d
        state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated


def convert_to_pytorch(path_to_checkpoint, model_state):
    with open(path_to_checkpoint, "rb") as f:
        caffe2_checkpoint = pickle.load(f, encoding="latin1")
    state_dict = OrderedDict()
    name_convert_func = get_name_convert_func()
    for key in caffe2_checkpoint["blobs"].keys():
        converted_key = name_convert_func(key)
        if converted_key in model_state:
            if caffe2_checkpoint["blobs"][key].shape == tuple(
                model_state[converted_key].shape
            ):
                state_dict[converted_key] = torch.tensor(
                    caffe2_checkpoint["blobs"][key]
                ).clone()
            else:
                logger.info(
                    "!! {}: {} does not match {}: {}".format(
                        key,
                        caffe2_checkpoint["blobs"][key].shape,
                        converted_key,
                        tuple(model_state[converted_key].shape),
                    )
                )
        else:
            assert any(
                prefix in key for prefix in ["momentum", "lr", "model_iter"]
            ), "{} can not be converted, got {}".format(key, converted_key)

    return state_dict


def init_from_pretrained(
    model,
    path_to_checkpoint,
    multi_gpu,
    dismissed_weights,
    caffe2_checkpoint=False,
    ignore_w=[],
):
    files = os.listdir(path_to_checkpoint) if os.path.exists(path_to_checkpoint) else []
    assert any("checkpoint" in f for f in files), "Checkpoint '{}' not found".format(
        path_to_checkpoint
    )
    name = sorted(files)[-1]
    checkpoint_path = os.path.join(path_to_checkpoint, name)
    if caffe2_checkpoint:
        model_load = model.module if multi_gpu else model
        model_state = convert_to_pytorch(checkpoint_path, model_load)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint["model_state"]
    logger.info(f"Loading pre-train model from {checkpoint_path}")
    for key in dismissed_weights:
        if key in model_state:
            del model_state[key]
    if multi_gpu:
        missing_keys, unexpected_keys = model.module.load_state_dict(
            model_state, strict=False
        )
    else:
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

    missed_keys = list(set(missing_keys) - set(dismissed_weights))
    if collections.Counter(missed_keys) == collections.Counter(ignore_w):
        return
    assert len(missed_keys) == 0, "Keys {} are missing in checkpoint.".format(
        missed_keys
    )
    assert (
        len(unexpected_keys) == 0
    ), "Unexpected keys found in checkpoint model state {}".format(unexpected_keys)


def filter_checkpoint_dict(model_state_dict, checkpoint_state_dict, dismissed_w):
    """
    Filtering checkpoint state dict based on the available modules in model
    :return:
    """
    updated_state_dict = {}
    for name, weight in model_state_dict.items():
        if name in checkpoint_state_dict:
            updated_state_dict[name] = checkpoint_state_dict[name]
        else:
            updated_state_dict[name] = weight
    for name in dismissed_w:
        if name in model_state_dict:
            updated_state_dict[name] = model_state_dict[name]

    return updated_state_dict


def load_checkpoint(
    path_to_checkpoint,
    model,
    fine_tune,
    num_gpus,
    dismissed_weights=[],
    optimizer=None,
    inflation=False,
    convert_from_caffe2=False,
):
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        fine_tune (bool): whether it is fine-tuning or not
        num_gpus (int): number of gpus
        dismissed_weights (list): name of modules to ignore when loading weights
        optimizer (optim): optimizer to load the historical state.
        inflation (bool): if True, inflate the weights from the checkpoint.
        convert_from_caffe2 (bool): if True, load the model from caffe2 and
            convert it to pytorch.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    assert os.path.exists(path_to_checkpoint), "Checkpoint '{}' not found".format(
        path_to_checkpoint
    )
    logger.info(f"Loading from {path_to_checkpoint}")
    model_load = model.module if num_gpus > 1 else model
    model_state_dict = model_load.state_dict()

    if convert_from_caffe2:
        checkpoint_model_state = convert_to_pytorch(path_to_checkpoint, model_state_dict)
        model_state = filter_checkpoint_dict(model_state_dict, checkpoint_model_state, dismissed_weights)

        if num_gpus > 1:
            missing_keys, unexpected_keys = model.module.load_state_dict(
                model_state, strict=False
            )
        else:
            missing_keys, unexpected_keys = model.load_state_dict(
                model_state, strict=False
            )
        # missed_keys = list(set(missing_keys) - set(dismissed_weights))
        assert len(missing_keys) == 0, "Keys {} are missing in checkpoint.".format(
            missing_keys
        )
        assert (
            len(unexpected_keys) == 0
        ), "Unexpected keys found in checkpoint model state {}".format(unexpected_keys)
        epoch = -1
    else:
        # Load the checkpoint on CPU to avoid GPU mem spike.
        checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
        checkpoint['model_state'] = filter_checkpoint_dict(model_state_dict,
                                                           checkpoint['model_state'],
                                                           dismissed_weights)

        if inflation:
            # Try to inflate the model.
            model_state_dict_3d = model.state_dict()
            inflated_model_dict = inflate_weight(
                checkpoint["model_state"], model_state_dict_3d
            )
            model.load_state_dict(inflated_model_dict)
        else:
            try:
                model.load_state_dict(checkpoint["model_state"])
            except Exception as e:
                # logger.info(f"Could not load model with error {e}")
                model.module.load_state_dict(checkpoint["model_state"])
            # Load the optimizer state (commonly not done when fine-tuning)
            if not fine_tune and optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state"])

        if "epoch" in checkpoint.keys() and not fine_tune:
            epoch = checkpoint["epoch"]
        else:
            epoch = -1
    return epoch
