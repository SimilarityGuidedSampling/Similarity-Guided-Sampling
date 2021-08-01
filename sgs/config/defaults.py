#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# BN epsilon.
_C.BN.EPSILON = 1e-5

# BN momentum.
_C.BN.MOMENTUM = 0.1

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# log weights of temporal sampling layer
_C.TRAIN.LOG_WEIGHTS = False

# fine tuning network for ucf101
_C.TRAIN.FINE_TUNE = False

# Dataset [kinetics, kineticsJpg].
_C.TRAIN.DATASET = "kinetics"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Number of steps to accumulate the gradients
_C.TRAIN.GRAD_ACCUMULATION_STEP = 1

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Experiment number to resume training from
_C.TRAIN.RESUME_EXPR_NUM = -1

# Path to the base 3D model checkpoint to load the initial weights.
_C.TRAIN.BASE_3D_CHECKPOINT_PATH = ""

# Init weights from a model that trained on other datasets (e.g. kinetics)
_C.TRAIN.INIT_FROM_PRETRAINED = False

# ignore given weights to be init randomly
_C.TRAIN.IGNORE_WEIGHTS = False

# Logging the output of fc in ts_layer in tensorboard
_C.TRAIN.LOG_FC_OUTPUT = False

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = False

# Dataset for testing.
_C.TEST.DATASET = "kinetics"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"

# to save activations from embedding for further analysis
_C.TEST.SAVE_EMBEDDING_ACT = False

# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

_C.RESNET.WIDTH_PER_GROUP_FAST = 8

_C.RESNET.WIDTH_PER_GROUP_SLOW = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]

# ---------------------------------------------------------------------------- #
# X3D  options
# See https://arxiv.org/abs/2004.04730 for details about X3D Networks.
# ---------------------------------------------------------------------------- #
_C.X3D = CfgNode()

# Width expansion factor.
_C.X3D.WIDTH_FACTOR = 1.0

# Depth expansion factor.
_C.X3D.DEPTH_FACTOR = 1.0

# Bottleneck expansion factor for the 3x3x3 conv.
_C.X3D.BOTTLENECK_FACTOR = 1.0  #

# Dimensions of the last linear layer before classificaiton.
_C.X3D.DIM_C5 = 2048

# Dimensions of the first 3x3 conv layer.
_C.X3D.DIM_C1 = 12

# Whether to scale the width of Res2, default is false.
_C.X3D.SCALE_RES2 = False

# Whether to use a BatchNorm (BN) layer before the classifier, default is false.
_C.X3D.BN_LIN5 = False

# Whether to use channelwise (=depthwise) convolution in the center (3x3x3)
# convolution operation of the residual blocks.
_C.X3D.CHANNELWISE_3x3x3 = True


# -----------------------------------------------------------------------------
# _X3D options, our implementation of X3D
# -----------------------------------------------------------------------------
_C._X3D = CfgNode()

# Transformation function.
_C._X3D.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C._X3D.NUM_GROUPS = 1

# The instance of X3D model currently ['S', 'M', 'XL']
_C._X3D.INSTANCE = "S"

# Apply relu in a inplace manner.
_C._X3D.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C._X3D.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C._X3D.ZERO_INIT_FINAL_BN = False

# Size of stride on different res stages.
_C._X3D.SPATIAL_STRIDES = [[2], [2], [2], [2]]

# Size of dilation on different res stages.
_C._X3D.SPATIAL_DILATIONS = [[1], [1], [1], [1]]

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C._X3D.NUM_BLOCK_TEMP_KERNEL = [[3], [5], [11], [7]]

# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

# -----------------------------------------------------------------------------
# Temporal Sampling options
# -----------------------------------------------------------------------------
_C.TEMPORAL_SAMPLING = CfgNode()

# to use order aware temporal sampling or not
_C.TEMPORAL_SAMPLING.ORDERED = False

# Number of bins for temporal sampling kernels
_C.TEMPORAL_SAMPLING.NUM_BINS = [-1]

# Type of distance to be used in temporal sampling
_C.TEMPORAL_SAMPLING.DISTANCE_TYPE = "L2"

# type of the similarity function from ['L1', 'L2', 'Mahl', 'Cosine']
_C.TEMPORAL_SAMPLING.SIMILARITY_FUNCTION_TYPE = "L2"

_C.TEMPORAL_SAMPLING.LATENT_DIMENSION = 3

# Type of sampling kernel to be used in temporal sampling ['linear', 'delta']
_C.TEMPORAL_SAMPLING.KERNEL_TYPE = "linear"

# Using ReLU in between fc layers
_C.TEMPORAL_SAMPLING.USE_RELU = True  # if False -> uses Swish

# Normalize the sampled temporal regions
_C.TEMPORAL_SAMPLING.KERNEL_NORM = True

# Drop frames during testing with batch_size = 1
_C.TEMPORAL_SAMPLING.DROP_ZERO_BINS = True

# scaling the gradient in backward function
_C.TEMPORAL_SAMPLING.GRAD_SCALE_VAL = 1.0

# SlowFast Pathway index for TS Layer
_C.TEMPORAL_SAMPLING.PATHWAYS = [0, 1]


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "sgs"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 400

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = [
    "3dresnet",
]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Randomly drop rate for Res-blocks, linearly increase from res2 to res5
_C.MODEL.DROPCONNECT_RATE = 0.0

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# the layer to apply temporal sampling [conv1, res2, res3, res4, res5, none]
_C.MODEL.TEMPORAL_LAYER_STAGE = "none"

# the layer to apply double temporal attnetion [conv1, res2, res3, res4, res5, none]
_C.MODEL.ATTENTION_LAYER_STAGE = "none"

# the layer to apply temporal pooling [conv1, res2, res3, res4, res5, none]
_C.MODEL.TEMPORAL_POOLING_LAYER_STAGE = "none"

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# to use depth wise convolution with temporal kernel size of 3
_C.MODEL.DEPTH_WISE_CONV = False

# list of module names to be ignore for fine-tuning
_C.MODEL.DISMISSED_WEIGHTS = []

# -----------------------------------------------------------------------------
# Slowfast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = "/Data/Kinetics400"

# Video path prefix if any.
_C.DATA.PATH_PREFIX = "/Data/Kinetics400"

# Json dataset meta info filename
_C.DATA.JSON_META_FILE = "kinetics200_info.json"

# The spatial crop size of the input clip.
_C.DATA.CROP_SIZE = 224

# The fraction of of clips per class to use for training
_C.DATA.DATA_FRACTION = 1.0

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# if True, sample uniformly in [1 / max_scalfe, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# A factor that will be multiplied to lr after lr_policy to
# have more control over lr during training
_C.SOLVER.ALPHA_FACTOR = 1.0

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./experiments"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = False

# Config file
_C.CONFIG_FILE = ""

# Distributed backend.
_C.DIST_BACKEND = "nccl"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


def _assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    assert cfg.TEST.NUM_SPATIAL_CROPS == 3

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
