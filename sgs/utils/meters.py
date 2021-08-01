#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
import matplotlib.pyplot as plt
from fvcore.common.timer import Timer

import sgs.utils.logging as logging
import sgs.utils.metrics as metrics
import sgs.utils.misc as misc

logger = logging.get_logger(__name__)


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(self, num_videos, num_clips, num_cls, overall_iters, writer):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
        """

        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))
        self.video_labels = torch.zeros((num_videos)).long()
        self.clip_count = torch.zeros((num_videos)).long()
        self.latest_video_idx = -1
        self.writer = writer
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            self.video_labels[vid_id] = labels[ind]
            self.video_preds[vid_id] += preds[ind]
            self.clip_count[vid_id] += 1

        self.latest_video_idx = vid_id

    def log_iter_stats(self, cur_iter, ks=(1, 5)):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
            ks
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        preds = self.video_preds[:self.latest_video_idx]
        lbls = self.video_labels[:self.latest_video_idx]
        num_topks_correct = metrics.topks_correct(
            preds, lbls, ks
        )
        topks = [
            (x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        assert len({len(ks), len(topks)}) == 1
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
            "num_videos": self.latest_video_idx
        }
        for k, topk in zip(ks, topks):
            stats["top{}_acc".format(k)] = "{:.{prec}f}".format(topk, prec=2)
            self.writer.add_scalar(f"test/top{k}_acc", topk, global_step=cur_iter)
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    self.clip_count, self.num_clips
                )
            )
            logger.warning(self.clip_count)

        num_topks_correct = metrics.topks_correct(
            self.video_preds, self.video_labels, ks
        )
        topks = [
            (x / self.video_preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        assert len({len(ks), len(topks)}) == 1
        stats = {"split": "test_final"}
        for k, topk in zip(ks, topks):
            stats["top{}_acc".format(k)] = "{:.{prec}f}".format(topk, prec=2)
        logging.log_json_stats(stats)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg, gloabl_step, writer):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        # tensorboard writer
        self.writer = writer
        self.global_iter = gloabl_step

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.loss.add_value(loss)
        self.lr = lr
        # Aggregate stats
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def log_stats_tensorboard(self):
        """
        log the stats of the current iteration using tensorboard summary writer.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        self.writer.add_scalar("train/loss",
                               self.loss.get_win_median(),
                               global_step=self.global_iter)
        self.writer.add_scalar("train/top1_acc",
                               100 - self.mb_top1_err.get_win_median(),
                               global_step=self.global_iter)
        self.writer.add_scalar("train/top5_acc",
                               100 - self.mb_top5_err.get_win_median().item(),
                               global_step=self.global_iter)
        self.global_iter += 1

    def log_fc_outputs(self, fc_out, name):
        self.writer.add_histogram(name,
                                  fc_out,
                                  global_step=self.global_iter)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "top1_err": self.mb_top1_err.get_win_median(),
            "top5_err": self.mb_top5_err.get_win_median(),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "top1_err": top1_err,
            "top5_err": top5_err,
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

        self.writer.add_scalar("train/lr", self.lr, cur_epoch)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg, writer=None):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

        self.writer = writer

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    # this is only for debug
    def log_frame_counts(self, activations, cur_epoch):
        for name, frame_dict in activations.items():
            total_num_frames = frame_dict['inp_num_frames']
            frame_active_masks = torch.cat(frame_dict["frame_mask"], dim=0)
            hist = torch.histc(frame_active_masks.float(), bins=total_num_frames, min=0, max=total_num_frames)
            fig = plt.figure(figsize=(6, 6))
            plt.bar(np.arange(total_num_frames), hist.numpy())
            plt.xlabel(f"#Bins(={total_num_frames})")
            plt.ylabel(f"#Clips(={frame_active_masks.shape[0]})")
            if self.writer:
                self.writer.add_figure(f"val/active_frame_histogram/{name}", fig, global_step=cur_epoch)
                plt.clf()
                plt.close(fig)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "top1_err": self.mb_top1_err.get_win_median(),
            "top5_err": self.mb_top5_err.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        self.min_top5_err = min(self.min_top5_err, top5_err)
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "top1_err": top1_err,
            "top5_err": top5_err,
            "min_top1_err": self.min_top1_err,
            "min_top5_err": self.min_top5_err,
            "mem": int(np.ceil(mem_usage)),
        }

        logging.log_json_stats(stats)

        if self.writer:
            self.writer.add_scalar("val/top1_acc", 100 - top1_err, global_step=cur_epoch)
            self.writer.add_scalar("val/top5_acc", 100 - top5_err, global_step=cur_epoch)

        return top1_err