#!/usr/bin/env python3

"""Video models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sgs.utils.weight_init_helper as init_helper
from sgs.models import head_helper, resnet_helper, stem_helper
import sgs.utils.logging as logging

logger = logging.get_logger(__name__)


# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}


# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "3dresnet": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "3dresnet": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}

_TEMPORAL_SAMPLINIG_STAGE = {
    "conv1": [True, False, False, False, False],
    "res2": [False, True, False, False, False],
    "res3": [False, False, True, False, False],
    "res4": [False, False, False, True, False],
    "res5": [False, False, False, False, True],
    "c1r2": [True, True, False, False, False],
    "c1r23": [True, True, True, False, False],
    "c1r234": [True, True, True, True, False],
    "none": [False] * 5,
    "all": [True] * 5,
}

_TEMPORAL_ATTENTION_STAGE = {
    "conv1": [True, False, False, False, False],
    "res2": [False, True, False, False, False],
    "res3": [False, False, True, False, False],
    "res4": [False, False, False, True, False],
    "res5": [False, False, False, False, True],
    "none": [False] * 5,
}


def pad_pathways(slow_pathway, fast_pathway):
    """

    :param slow_pathway:
    :param fast_pathway:
    :return:
    """
    temporal_x_s = slow_pathway.shape[2]
    temporal_fuse = fast_pathway.shape[2]
    out_temporal = max(temporal_x_s, temporal_fuse)
    pad_size_fuse = out_temporal - temporal_fuse
    pad_size_s = out_temporal - temporal_x_s
    if pad_size_s > 0:
        pad_start = pad_size_s // 2
        pad_end = pad_size_s - pad_start
        slow_pathway = F.pad(
            slow_pathway, (0, 0, 0, 0, pad_start, pad_end), mode="constant", value=0
        )
    if pad_size_fuse > 0:
        pad_start = pad_size_fuse // 2
        pad_end = pad_size_fuse - pad_start
        fast_pathway = F.pad(
            fast_pathway, (0, 0, 0, 0, pad_start, pad_end), mode="constant", value=0
        )

    return slow_pathway, fast_pathway


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = nn.BatchNorm3d(
            dim_in * fusion_conv_channel_ratio, eps=eps, momentum=bn_mmt
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s, fuse = pad_pathways(x_s, fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


class SlowFastModel(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFastModel, self).__init__()
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        assert (
            sum(_TEMPORAL_SAMPLINIG_STAGE[cfg.MODEL.TEMPORAL_LAYER_STAGE]) <= 1
        ), "Applying temporal sampling to more than one stage"
        (conv1, ts2, ts3, ts4, ts5) = _TEMPORAL_SAMPLINIG_STAGE[
            cfg.MODEL.TEMPORAL_LAYER_STAGE
        ]
        logger.info(
            f"Applying Temporal Sampling to {cfg.MODEL.TEMPORAL_LAYER_STAGE} stage {_TEMPORAL_SAMPLINIG_STAGE[cfg.MODEL.TEMPORAL_LAYER_STAGE]}"
        )

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            num_frames=cfg.DATA.NUM_FRAMES,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            num_frames=cfg.DATA.NUM_FRAMES,
            alpha=cfg.SLOWFAST.ALPHA,
            use_temporal_sampling=ts2,
            temporal_sampling_pathways=cfg.TEMPORAL_SAMPLING.PATHWAYS,
            temporal_sampling_params=cfg.TEMPORAL_SAMPLING,
        )

        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            num_frames=cfg.DATA.NUM_FRAMES,
            alpha=cfg.SLOWFAST.ALPHA,
            use_temporal_sampling=ts3,
            temporal_sampling_params=cfg.TEMPORAL_SAMPLING,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            num_frames=cfg.DATA.NUM_FRAMES,
            alpha=cfg.SLOWFAST.ALPHA,
            use_temporal_sampling=ts4,
            temporal_sampling_params=cfg.TEMPORAL_SAMPLING,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            num_frames=cfg.DATA.NUM_FRAMES,
            alpha=cfg.SLOWFAST.ALPHA,
            use_temporal_sampling=ts5,
            temporal_sampling_params=cfg.TEMPORAL_SAMPLING,
        )

        self.head = head_helper.ResNetAdaptHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            num_pathways=self.num_pathways,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
        )

    def forward(self, x):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        x = self.head(x)
        return x


class ResNet3D(nn.Module):
    """"""

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet3D, self).__init__()
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]
        assert (
            sum(_TEMPORAL_SAMPLINIG_STAGE[cfg.MODEL.TEMPORAL_LAYER_STAGE]) <= 1
        ), "Applying temporal sampling to more than one stage"
        (conv1, ts2, ts3, ts4, ts5) = _TEMPORAL_SAMPLINIG_STAGE[
            cfg.MODEL.TEMPORAL_LAYER_STAGE
        ]
        logger.info(
            f"Applying Temporal Sampling to {cfg.MODEL.TEMPORAL_LAYER_STAGE} stage {_TEMPORAL_SAMPLINIG_STAGE[cfg.MODEL.TEMPORAL_LAYER_STAGE]}"
        )

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP_SLOW
        width_per_group_fast = cfg.RESNET.WIDTH_PER_GROUP_FAST
        dim_inner = num_groups * width_per_group // 2
        dim_inner_fast = num_groups * width_per_group_fast

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group_fast],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            num_frames=cfg.DATA.NUM_FRAMES,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group_fast],
            dim_out=[width_per_group_fast * 4],
            dim_inner=[dim_inner_fast],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            num_frames=cfg.DATA.NUM_FRAMES,
            use_temporal_sampling=ts2,
            temporal_sampling_params=cfg.TEMPORAL_SAMPLING,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group_fast * 4],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            num_frames=cfg.DATA.NUM_FRAMES,
            use_temporal_sampling=ts3,
            temporal_sampling_params=cfg.TEMPORAL_SAMPLING,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            num_frames=cfg.DATA.NUM_FRAMES,
            use_temporal_sampling=ts4,
            temporal_sampling_params=cfg.TEMPORAL_SAMPLING,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            num_frames=cfg.DATA.NUM_FRAMES,
            use_temporal_sampling=ts5,
            temporal_sampling_params=cfg.TEMPORAL_SAMPLING,
        )

        self.head = head_helper.ResNetAdaptHead(
            dim_in=[width_per_group * 16],
            num_classes=cfg.MODEL.NUM_CLASSES,
            num_pathways=self.num_pathways,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
        )

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.head(x)
        return x