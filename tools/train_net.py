#!/usr/bin/env python3

"""Train a video classification model."""
import os
from functools import partial

import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import update_bn_stats, get_bn_modules

import sgs.models.losses as losses
import sgs.models.optimizer as optim
import sgs.utils.checkpoint as cu
import sgs.utils.logging as logging
import sgs.utils.metrics as metrics
import sgs.utils.misc as misc
from sgs.datasets import loader
from sgs.models import model_builder
from sgs.utils.meters import TrainMeter, ValMeter
from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, device, writer
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            sgs/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, (inputs, labels, _, meta) in tqdm(
        enumerate(train_loader), total=data_size
    ):

        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(device)
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr, alpha=cfg.SOLVER.ALPHA_FACTOR)

        # Perform the forward pass.
        preds = model(inputs)

        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # Compute the loss.
        loss = loss_fun(preds, labels)

        # check Nan Loss.
        misc.check_nan_losses(loss, model, optimizer, cfg, cur_epoch, cur_iter)
        loss = loss / cfg.TRAIN.GRAD_ACCUMULATION_STEP

        glob_step = (cur_epoch * len(train_loader)) + cur_iter
        if cfg.TRAIN.LOG_WEIGHTS:
            train_meter.log_weights(model, glob_step, cfg.NUM_GPUS > 1)

        loss.backward()

        if (cur_iter + 1) % cfg.TRAIN.GRAD_ACCUMULATION_STEP == 0:
            # Update the parameters.
            optimizer.step()
            # Zero grad.
            optimizer.zero_grad()

            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]

            # Copy the stats from GPU to CPU (sync point).
            loss, top1_err, top5_err = (
                loss.item() * cfg.TRAIN.GRAD_ACCUMULATION_STEP,
                top1_err.item(),
                top5_err.item(),
            )
            train_meter.iter_toc()

            # Update and log stats.
            train_meter.update_stats(top1_err, top5_err, loss, lr, inputs[0].size(0))

            train_meter.log_stats_tensorboard()
            train_meter.log_iter_stats(cur_epoch, cur_iter)
            train_meter.iter_tic()

    # Update the parameters.
    optimizer.step()
    # Perform the backward pass.
    optimizer.zero_grad()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, device):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            sgs/config/defaults.py
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    logger.info(f"Validation Started")

    hook_handles = []
    if cfg.NUM_GPUS == 1:
        module_names_dict = dict(model.named_modules())
    else:
        module_names_dict = dict(model.module.named_modules())

    activations = {}
    sampler_modules = []
    if 's2.pathway1_ts_layer' in module_names_dict:
        sampler_modules.append((module_names_dict['s2.pathway1_ts_layer'], "fast_pathway"))
        activations['fast_pathway'] =  {"frame_mask": []}
    if 's2.pathway0_ts_layer' in module_names_dict:
        sampler_modules.append((module_names_dict['s2.pathway0_ts_layer'], "slow_pathway"))
        activations['slow_pathway'] = {"frame_mask": []}
    if 's2.ts_layer' in module_names_dict:
        sampler_modules.append((module_names_dict['s2.ts_layer'], "single_pathway_model"))
        activations['single_pathway_model'] = {"frame_mask": []}

    def save_t_sampler(name, mod, inp, out):
        ts_out, ts_mask = out
        # number of frames before t_sampler module
        t = inp[0].shape[2]
        if t == cfg.DATA.NUM_FRAMES:
            activations[name]['frame_mask'].append(ts_mask.sum(1).cpu())
            activations[name]['inp_num_frames'] = t
        else:
            activations[name]['frame_mask'].append(ts_mask.sum(1).cpu())
            activations[name]['inp_num_frames'] = t

    if len(sampler_modules) > 0:
        for m, name in sampler_modules:
            hook_handles.append(m.register_forward_hook(partial(save_t_sampler, name)))

    for cur_iter, (inputs, labels, _, meta) in tqdm(
        enumerate(val_loader), total=len(val_loader)
    ):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(device)
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)

        preds = model(inputs)

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

        # Combine the errors across the GPUs.
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]

        # Copy the errors from GPU to CPU (sync point).
        top1_err, top5_err = top1_err.item(), top5_err.item()

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(top1_err, top5_err, inputs[0].size(0))

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    if len(activations) != 0:
        val_meter.log_frame_counts(activations, cur_epoch)
    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    for handle in hook_handles:
        handle.remove()
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, device, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        logger.info("Updating BN stats")
        for inputs, _, _, _ in tqdm(loader, total=len(loader)):
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(device)
            else:
                inputs = inputs.to(device)
            yield inputs

    if len(loader) < num_iters:
        num_iters = len(loader) // 2

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            sgs/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.EXPR_NUM)

    cfg.LOG_PERIOD = int(np.lcm(cfg.TRAIN.GRAD_ACCUMULATION_STEP, cfg.LOG_PERIOD))

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)
    if cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, is_train=True)

    if cfg.NUM_GPUS > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # Transfer the model to device(s)
    model = model.to(device)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(
        cfg.OUTPUT_DIR, cfg.TRAIN.RESUME_EXPR_NUM
    ):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(
            cfg.OUTPUT_DIR, cfg.TRAIN.RESUME_EXPR_NUM
        )
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint,
            model=model,
            fine_tune=False,
            num_gpus=cfg.NUM_GPUS,
            optimizer=optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.TRAIN.FINE_TUNE,
            cfg.NUM_GPUS,
            dismissed_weights=cfg.MODEL.DISMISSED_WEIGHTS,
            optimizer=optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0
        logger.info("Training from scratch")

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create tensorboard summary writer
    writer = SummaryWriter(cfg.SUMMARY_PATH, flush_secs=15)

    # Create meters.
    train_meter = TrainMeter(
        len(train_loader), cfg, start_epoch * (len(train_loader)), writer
    )
    val_meter = ValMeter(len(val_loader), cfg, writer)

    # Print summary path.
    logger.info("Summary path {}".format(cfg.SUMMARY_PATH))

    logger.info("Process PID {}".format(os.getpid()))
    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, device, writer
        )

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                train_loader, model, device, cfg.BN.NUM_BATCHES_PRECISE
            )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            logger.info("Saving checkpoint")
            cu.save_checkpoint(
                cfg.OUTPUT_DIR, cfg.EXPR_NUM, model, optimizer, cur_epoch, cfg
            )

        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, device)
