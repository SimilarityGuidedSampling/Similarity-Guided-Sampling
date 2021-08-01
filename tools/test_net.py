#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import os
import numpy as np
import pprint

import torch
from torch.utils.tensorboard import SummaryWriter

import sgs.utils.checkpoint as cu
import sgs.utils.logging as logging
import sgs.utils.misc as misc
from sgs.datasets import loader
from sgs.models import model_builder
from sgs.utils.meters import TestMeter

from tqdm import tqdm


logger = logging.get_logger(__name__)

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.

    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            sgs/config/defaults.py
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Transfer the data to the current GPU device.
        labels = labels.cuda()
        video_idx = video_idx.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        # Perform the forward pass.
        preds = model(inputs)

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            preds.detach().cpu(),
            labels.detach().cpu(),
            video_idx.detach().cpu(),
        )

        if (cur_iter+1) % 20 == 0:
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()


    # Log epoch stats and print the final testing results.
    test_meter.finalize_metrics()
    test_meter.reset()


def test(cfg):
    """
    Evaluate the model on the val set.
    :param cfg: (CfgNode) configs. Details can be found in
            sgs/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "test")
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.EXPR_NUM)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)
    misc.log_model_info(model, cfg, is_train=False)

    if cfg.NUM_GPUS > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # Transfer the model to device(s)
    model = model.to(device)

    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        files = os.listdir(cfg.TEST.CHECKPOINT_FILE_PATH) if os.path.exists(cfg.TEST.CHECKPOINT_FILE_PATH) else []
        assert any("checkpoint" in f for f in files), "Checkpoint '{}' not found".format(cfg.TEST.CHECKPOINT_FILE_PATH)
        name = sorted(files)[-1]
        checkpoint_path = os.path.join(cfg.TEST.CHECKPOINT_FILE_PATH, name)
        logger.info(f"Loading model for test from {cfg.TEST.CHECKPOINT_FILE_PATH}")
        cu.load_checkpoint(
            checkpoint_path,
            model,
            num_gpus=cfg.NUM_GPUS,
            fine_tune=False,
        )
    elif cu.has_checkpoint(cfg.OUTPUT_DIR, cfg.EXPR_NUM):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        # raise NotImplementedError("Unknown way to load checkpoint.")
        logger.info("Testing with random initialization. Only for debugging.")

    cfg.SUMMARY_PATH = os.path.join(cfg.SUMMARY_PATH, "test")
    # Create tensorboard summary writer
    writer = SummaryWriter(cfg.SUMMARY_PATH, flush_secs=15)

    # Create the video val loader.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Summary path {}".format(cfg.SUMMARY_PATH))
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        len(test_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        writer
    )

    # # Perform multi-view test on the entire dataset.
    perform_test(test_loader, model, test_meter, cfg)