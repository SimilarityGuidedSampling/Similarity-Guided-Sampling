#!/usr/bin/env python3

"""Wrapper to train and test a video classification model."""

import os
import sys
from os.path import join, split, splitext
import argparse

import sgs.utils.checkpoint as cu
from sgs.config.defaults import get_cfg

import sgs.utils.misc as misc

from test_net import test
from train_net import train

import json


def parse_args():
    """
    Parse the following arguments for the video training and testing pipeline.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/3DResNet50+ATFR.yaml",  # Dev_SLOWFAST_8x8_R50_K20.yaml
        type=str,
    )
    parser.add_argument(
        "--resume_expr_num",
        help="The experiment number to resume training",
        default=1,
        type=int,
    )
    parser.add_argument(
        "opts",
        help="See sgs/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "resume_expr_num"):
        cfg.RESUME_EXPR_NUM = args.resume_expr_num
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    cfg.CONFIG_FILE = args.cfg_file
    cfg_file_name = splitext(split(args.cfg_file)[1])[0]
    cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, cfg_file_name)

    return cfg


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    summary_path = misc.check_path(join(cfg.OUTPUT_DIR, "summary"))
    cfg.EXPR_NUM = str(misc.find_latest_experiment(join(cfg.OUTPUT_DIR, "summary")) + 1)
    if cfg.TRAIN.AUTO_RESUME and cfg.TRAIN.RESUME_EXPR_NUM > 0:
        cfg.EXPR_NUM = cfg.TRAIN.RESUME_EXPR_NUM
    cfg.SUMMARY_PATH = misc.check_path(join(summary_path, "{}".format(cfg.EXPR_NUM)))
    cfg.CONFIG_LOG_PATH = misc.check_path(
        join(cfg.OUTPUT_DIR, "config", "{}".format(cfg.EXPR_NUM))
    )
    with open(os.path.join(cfg.CONFIG_LOG_PATH, "config.yaml"), "w") as json_file:
        json.dump(cfg, json_file, indent=2)
    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR, cfg.EXPR_NUM)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        train(cfg=cfg)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        test(cfg=cfg)


if __name__ == "__main__":
    main()
