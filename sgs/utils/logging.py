#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""
from os import makedirs
from os.path import join
import builtins
import decimal
import logging
import sys
import simplejson



def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


def setup_logging(output_dir, expr_num):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
    log_folder = join(output_dir, "logs")
    makedirs(log_folder, exist_ok=True)
    log_file = join(log_folder, f"expr_{expr_num}.log")
    # Enable logging for the master process.

    fh = logging.FileHandler(filename=log_file, mode='a')
    ch = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, format=_FORMAT, handlers=[fh, ch])


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def log_json_stats(stats):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.6f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("json_stats: {:s}".format(json_stats))
