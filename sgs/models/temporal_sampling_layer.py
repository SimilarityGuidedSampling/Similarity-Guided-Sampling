import torch.nn as nn
from torch import Tensor
from torch.nn.functional import adaptive_avg_pool3d

from sgs.models.TemporalSampler import TemporalSampling
from sgs.models.operators import Swish


class TSLayer(nn.Module):
    def __init__(self, num_channels, num_frames, num_bins, temporal_sampling_params):
        """
            Temporal Sampling Layer, applies temporal sampling operation over the input feature map
        :param num_channels:
        :param num_frames:
        :param num_bins:
        :param temporal_sampling_params
        """
        super(TSLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(True) if temporal_sampling_params.USE_RELU else Swish(),
            nn.Linear(num_channels // 2, temporal_sampling_params.LATENT_DIMENSION),
        )
        self.bn = nn.BatchNorm1d(temporal_sampling_params.LATENT_DIMENSION)
        if isinstance(num_bins, list):
            num_bins = num_bins[0]
        self.t_sampler = TemporalSampling(
            distance_type=temporal_sampling_params.DISTANCE_TYPE,
            kernel_type=temporal_sampling_params.KERNEL_TYPE,
            num_bins=num_bins,
            grad_scale_val=temporal_sampling_params.GRAD_SCALE_VAL,
            drop_bins=temporal_sampling_params.DROP_ZERO_BINS,
        )

    def forward(self, input: Tensor) -> Tensor:
        """

        :param input: feature map of size [N, C, T, H, W]
        :return:
        """
        n, c, t, _, _ = input.shape
        feature_down = adaptive_avg_pool3d(input, (t, 1, 1)).view(n, c, t)  # [N, C, T]
        temporal_features = feature_down.transpose(1, 2).view(n, t, -1)  # [N, T, C]
        fc_out = self.fc(temporal_features)  # [N, T, LD]
        fc_out = fc_out.transpose(1, 2)  # [N, LD, T]
        fc_out = self.bn(fc_out)

        ts_out = self.t_sampler(input=input, embeddings=fc_out)

        return ts_out
