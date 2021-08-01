#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import os
from datetime import datetime
import torch
from fvcore.nn.flop_count import flop_count
from torch import nn

import sgs.utils.logging as logging
from sgs.utils.checkpoint import get_checkpoint_dir
from sgs.datasets.utils import pack_pathway_output

import matplotlib.pyplot as plt

logger = logging.get_logger(__name__)


def check_nan_losses(loss, model, optimizer, cfg, cur_epoch, cur_iter):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        saving_model = model.module if cfg.NUM_GPUS > 1 else model
        os.makedirs(get_checkpoint_dir(cfg.OUTPUT_DIR, cfg.EXPR_NUM), exist_ok=True)
        name = "nan_loss_checkpoint_epoch_{:05d}.pyth".format(cur_epoch+1)
        path_to_checkpoint = os.path.join(get_checkpoint_dir(cfg.OUTPUT_DIR, cfg.EXPR_NUM), name)
        checkpoint = {
            "epoch": cur_epoch,
            "iter": cur_iter,
            "optimizer_state": optimizer.state_dict(),
            "model_state": saving_model.state_dict(),
            "cfg": cfg.dump(),
        }
        torch.save(checkpoint, path_to_checkpoint)
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def _get_model_analysis_input(cfg, use_train_input):
    """
    Return a dummy input for model analysis with batch size 1. The input is
        used for analyzing the model (counting flops and activations etc.).
    Args:
        cfg (CfgNode): configs. Details can be found in
            sgs/config/defaults.py
        use_train_input (bool): if True, return the input for training. Otherwise,
            return the input for testing.
    Returns:
        inputs: the input for model analysis.
    """
    rgb_dimension = 3
    if use_train_input:
        input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.NUM_FRAMES,
            cfg.DATA.TRAIN_CROP_SIZE,
            cfg.DATA.TRAIN_CROP_SIZE,
        )
    else:
        input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.NUM_FRAMES,
            cfg.DATA.TEST_CROP_SIZE,
            cfg.DATA.TEST_CROP_SIZE,
        )
    model_inputs = pack_pathway_output(cfg, input_tensors)
    for i in range(len(model_inputs)):
        model_inputs[i] = model_inputs[i].unsqueeze(0)
        if cfg.NUM_GPUS:
            model_inputs[i] = model_inputs[i].cuda(non_blocking=True)


    inputs = (model_inputs,)
    return inputs


def get_flop_stats(model, cfg, is_train):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            sgs/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.

    Returns:
        float: the total number of gflops of the given model.
    """
    model_mode = model.training
    model.eval()
    inputs = _get_model_analysis_input(cfg, is_train)
    model = model.to('cuda')
    gflop_dict, _ = flop_count(model, inputs)
    gflops = sum(gflop_dict.values())
    model.train(model_mode)
    return gflops


def log_model_info(model, cfg, is_train=True):
    """
    Log info, includes number of parameters, gpu usage and gflops.
    Args:
        model (model): model to log the info.
        cfg (CfgNode): configs. Details can be found in
            sgs/config/defaults.py
        is_train (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))

    logger.info(
        "FLOPs: {:,} GFLOPs".format(get_flop_stats(model, cfg, is_train))
    )
    logger.info("Config file name: {}".format(cfg.CONFIG_FILE))
    if cfg.MODEL.TEMPORAL_LAYER_STAGE != "none":
        num_bins = cfg.TEMPORAL_SAMPLING.NUM_BINS
        if num_bins == -1:
            num_bins = cfg.DATA.NUM_FRAMES
        logger.info("Num of Bins: {}".format(num_bins))

def is_eval_epoch(cfg, cur_epoch):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            sgs/config/defaults.py
        cur_epoch (int): current epoch.
    """
    return (
        cur_epoch + 1
    ) % cfg.TRAIN.EVAL_PERIOD == 0 or cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH


def plot_input(tensor, bboxes=(), texts=(), path="./tmp_vis.png"):
    """
    Plot the input tensor with the optional bounding box and save it to disk.
    Args:
        tensor (tensor): a tensor with shape of `NxCxHxW`.
        bboxes (tuple): bounding boxes with format of [[x, y, h, w]].
        texts (tuple): a tuple of string to plot.
        path (str): path to the image to save to.
    """
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    f, ax = plt.subplots(nrows=1, ncols=tensor.shape[0], figsize=(50, 20))
    for i in range(tensor.shape[0]):
        ax[i].axis("off")
        ax[i].imshow(tensor[i].permute(1, 2, 0))
        # ax[1][0].axis('off')
        if bboxes is not None and len(bboxes) > i:
            for box in bboxes[i]:
                x1, y1, x2, y2 = box
                ax[i].vlines(x1, y1, y2, colors="g", linestyles="solid")
                ax[i].vlines(x2, y1, y2, colors="g", linestyles="solid")
                ax[i].hlines(y1, x1, x2, colors="g", linestyles="solid")
                ax[i].hlines(y2, x1, x2, colors="g", linestyles="solid")

        if texts is not None and len(texts) > i:
            ax[i].text(0, 0, texts[i])
    f.savefig(path)


def frozen_bn_stats(model):
    """
    Set all the bn layers to eval mode.
    Args:
        model (model): model to set bn layers to eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()


def find_latest_experiment(path):
    list_of_experiments = os.listdir(path)
    list_of_int_experiments = []
    for exp in list_of_experiments:
        try:
            int_exp = int(exp)
        except ValueError:
            continue
        list_of_int_experiments.append(int_exp)

    if len(list_of_int_experiments) == 0:
        return 0

    return max(list_of_int_experiments)


def check_path(path):
    os.makedirs(path, exist_ok=True)
    return path
