#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t 3D stem helper."""

import torch.nn as nn

from sgs.models.temporal_sampling_layer import TSLayer
from sgs.models.attention import DoubleAttentionLayer
import sgs.utils.logging as logging

logger = logging.get_logger(__name__)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_stem_func(name):
    """
    Retrieves the stem module by name.
    """
    trans_funcs = {"x3d_stem": X3DStem, "basic_stem": ResNetBasicStem}
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class VideoModelStem(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        num_frames,
        alpha=-1,
        x3d_stem=False,
        inplace_relu=True,
        num_groups=[1, 1],
        dwise_conv=False,
        pool_layer=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        stem_func_name="basic_stem",
        use_temporal_sampling=False,
        temporal_sampling_params={},
        use_temporal_attention=False,
        use_temporal_pooling=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, SlowOnly
        and etc), list size of 2 for two pathway models (SlowFast).

        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            pool_layer (bool): whether to use pooling layer or not
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
            default is nn.BatchNorm3d.
            stem_func_name (string): name of the the stem function applied on
                input to the network.
        """
        super(VideoModelStem, self).__init__()

        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(kernel),
                    len(stride),
                    len(padding),
                }
            )
            == 1
        ), "Input pathway dimensions are not consistent."
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.num_groups = num_groups
        self.pool_layer = pool_layer
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.norm_module = norm_module
        self.depth_wise = dwise_conv
        self.x3d_stem = x3d_stem
        self.temporal_sampling_params = temporal_sampling_params
        self.use_temporal_sampling = use_temporal_sampling
        self.num_frames = num_frames
        self.alpha = alpha

        self.use_temporal_attention = use_temporal_attention
        self.use_temporal_pooling = use_temporal_pooling

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, stem_func_name)

    def _construct_stem(self, dim_in, dim_out, stem_func_name):
        trans_func = get_stem_func(stem_func_name)
        num_frames = -1
        num_bins = -1
        if self.use_temporal_sampling:
            if len(dim_in) == 2:
                assert self.alpha > 0, "Factor alpha sgs is not given"
                num_frames = [self.num_frames // self.alpha, self.num_frames]
                num_bins = self.temporal_sampling_params.NUM_BINS
                assert (
                    len(num_bins) == 2
                ), "NUM_BINS for both two pathways should be provided."
            else:
                num_frames = [self.num_frames]
                num_bins = self.temporal_sampling_params.NUM_BINS
                for idx in range(len(num_bins)):
                    if num_bins[idx] == -1:
                        num_bins[idx] = self.num_frames

        for pathway in range(len(dim_in)):
            stem = trans_func(
                dim_in[pathway],
                dim_out[pathway],
                self.kernel[pathway],
                self.stride[pathway],
                self.padding[pathway],
                self.inplace_relu,
                self.eps,
                self.bn_mmt,
                self.norm_module,
                num_frames=num_frames,
                num_bins=num_bins,
                use_temporal_sampling=self.use_temporal_sampling,
                temporal_sampling_params=self.temporal_sampling_params,
                use_temporal_attention=self.use_temporal_attention,
                use_temporal_pooling=self.use_temporal_pooling,
            )
            self.add_module("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        assert (
            len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(x[pathway])
        return x


class ResNetBasicStem(nn.Module):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        num_frames=-1,
        num_bins=-1,
        use_temporal_sampling=False,
        temporal_sampling_params={},
        use_temporal_attention=False,
        use_temporal_pooling=False,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.bn = norm_module(num_features=dim_out, eps=self.eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)
        self.pool_layer = nn.MaxPool3d(
            kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool_layer(x)
        return x


class X3DStem(nn.Module):
    """
    X3D's 3D stem module.
    Performs a spatial followed by a depthwise temporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        num_frames=-1,
        num_bins=-1,
        use_temporal_sampling=False,
        temporal_sampling_params={},
        use_temporal_attention=False,
        use_temporal_pooling=False,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(X3DStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.temporal_sampling_params = temporal_sampling_params
        self.use_temporal_sampling = use_temporal_sampling
        self.num_frames = num_frames
        self.num_bins = num_bins
        self.use_temporal_attention = use_temporal_attention
        self.use_temporal_pooling = use_temporal_pooling

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv_xy = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=(1, self.kernel[1], self.kernel[2]),
            stride=(1, self.stride[1], self.stride[2]),
            padding=(0, self.padding[1], self.padding[2]),
            bias=False,
        )
        self.conv = nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=(self.kernel[0], 1, 1),
            stride=(self.stride[0], 1, 1),
            padding=(self.padding[0], 0, 0),
            bias=False,
            groups=dim_out,
        )

        self.bn = norm_module(num_features=dim_out, eps=self.eps, momentum=self.bn_mmt)
        if self.use_temporal_sampling:
            logger.info("Using Temporal Sampling Layer")
            self.ts_layer = TSLayer(
                num_channels=dim_out,
                num_frames=self.num_frames[0],
                num_bins=self.num_bins,
                temporal_sampling_params=self.temporal_sampling_params,
            )
        elif self.use_temporal_attention:
            logger.info("Using Double attention")
            self.attn_layer = DoubleAttentionLayer(in_channels=dim_out)
        elif self.use_temporal_pooling:
            logger.info("Using Average Temporal Pooling")
            self.temporal_pooling_layer = nn.AvgPool3d(kernel_size=(2, 1, 1))

        self.relu = nn.ReLU(self.inplace_relu)

    def forward(self, x):
        x = self.conv_xy(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.use_temporal_sampling:
            x, _ = self.ts_layer(x)
        elif self.use_temporal_attention:
            x = self.attn_layer(x)
        elif self.use_temporal_pooling:
            x = self.temporal_pooling_layer(x)

        x = self.relu(x)
        return x


class ResNetX3DBasicStem(nn.Module):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        num_groups=1,
        pool_layer=True,
        eps=1e-5,
        bn_mmt=0.1,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            pool_layer (bool): whether to use pooling layer or not
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(ResNetX3DBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.groups = num_groups
        self.inplace_relu = inplace_relu
        self.use_pool_layer = pool_layer
        self.eps = eps
        self.bn_mmt = bn_mmt

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        num_groups = self.groups
        self.conv_bn_relu = nn.Sequential()
        if dim_out % self.groups != 0:
            new_dim_in = _make_divisible(dim_in, 4)
            pre_conv = nn.Conv3d(
                dim_in,
                new_dim_in,
                kernel_size=1,
                stride=1,
                bias=False,
            )
            self.conv_bn_relu.add_module("pre_conv", pre_conv)
            self.conv_bn_relu.add_module(
                "pre_conv_bn",
                nn.BatchNorm3d(new_dim_in, eps=self.eps, momentum=self.bn_mmt),
            )
            self.conv_bn_relu.add_module("pre_conv_relu", nn.ReLU(self.inplace_relu))
            num_groups = new_dim_in
            dim_in = new_dim_in

        conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
            groups=num_groups,
        )
        bn = nn.BatchNorm3d(dim_out, eps=self.eps, momentum=self.bn_mmt)
        relu = nn.ReLU(self.inplace_relu)
        self.conv_bn_relu.add_module("conv", conv)
        self.conv_bn_relu.add_module("bn", bn)
        self.conv_bn_relu.add_module("relu", relu)

        if self.use_pool_layer:
            self.pool_layer = nn.MaxPool3d(
                kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
            )

    def forward(self, x):
        x = self.conv_bn_relu(x)
        if self.use_pool_layer:
            x = self.pool_layer(x)
        return x


class X3DBasiConv(nn.Module):
    """
    Output conv of X3D 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        num_groups=1,
        eps=1e-5,
        bn_mmt=0.1,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(X3DBasiConv, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.groups = num_groups
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
            groups=self.groups,
        )
        self.bn = nn.BatchNorm3d(dim_out, eps=self.eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)

    def forward(self, x):
        x = x[0]
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return [x]
