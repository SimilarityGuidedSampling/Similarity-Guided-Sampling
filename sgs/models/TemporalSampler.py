from typing import Tuple
import torch
import torch.nn as nn

from torch import Tensor
import sgs.utils.logging as logging

from sgs.models.distance_functions import _DISTANCE_FUNCTION_TYPES
from sgs.models.sampling_kernels import _SAMPLING_KERNELS_TYPES

logger = logging.get_logger(__name__)


class TemporalSampling(nn.Module):

    def __init__(
        self,
        distance_type: str,
        kernel_type: str,
        num_bins: int = -1,
        grad_scale_val: float = 1.0,
        drop_bins: bool = False,
    ):
        """

        :param distance_type:
        :param kernel_type:
        :param num_bins: if -1 then num_bins = T
        :param grad_scale_val: the value to scale the gradient
        :param drop_bins: whether to drop zero bins or not
        """
        super(TemporalSampling, self).__init__()
        self.num_bins = num_bins

        if distance_type in _DISTANCE_FUNCTION_TYPES.keys():
            self.distance_function = _DISTANCE_FUNCTION_TYPES[distance_type]
        else:
            raise Exception("Invalid distance type name (%s)" % distance_type)

        if kernel_type in _SAMPLING_KERNELS_TYPES.keys():
            self.sampling_kernel = _SAMPLING_KERNELS_TYPES[kernel_type](
                grad_scale_val=grad_scale_val
            )
        else:
            raise Exception("Invalid kernel type name (%s)" % kernel_type)

        self.drop_zero_bins = drop_bins
        if drop_bins:
            logger.info("Dropping zero bins")


    @staticmethod
    def create_bins(start: Tensor, end: Tensor, steps: int) -> Tuple:
        """
        :param start: [N x 1]
        :param end: [N x 1]
        :param steps: num steps
        :return: Bins means tensor [N x steps]
        """
        domain_size = end - start  # [N x 1]
        steps_size = domain_size / steps  # [N x 1]
        mu_start = start + steps_size / 2  # [N x 1]
        index_tensor = (
            torch.arange(start=0, end=steps).view(1, -1).expand(start.shape[0], -1)
        )  # [N x steps]
        index_tensor = index_tensor.to(steps_size.device)
        step_tensor = index_tensor * steps_size  # [N x steps]
        bins_mu = step_tensor + mu_start  # [N x steps]
        return bins_mu, steps_size

    @staticmethod
    def normalize_distances(dist: Tensor) -> Tensor:
        """

        :param dist: [N x T]
        :return: [N x T]
            distances normalized for each batch between 0 and 1
        """
        min_distances = dist.min(dim=1)[0].view(-1, 1)  # [N x 1]
        max_distances = dist.max(dim=1)[0].view(-1, 1)  # [N x 1]
        dist_norm = (dist - min_distances) / (max_distances - min_distances)
        return dist_norm

    def forward(self , input: Tensor , embeddings: Tensor) -> Tuple:
        """

        :param input: [N x C x T x H x W]
        :param embeddings: [N x K x T]
        :return:
        """
        N, C, T, H, W = input.shape
        if self.num_bins == -1:
            num_bins = T  # B
        else:
            num_bins = self.num_bins  # B

        # calculating distances using sampling params ----
        distances = self.distance_function(embeddings)  # [N x T]

        # normalizing the distances ----
        max_distances = distances.max(dim=1)[0].view(-1, 1)  # [N x 1]
        distances = distances * (2 * num_bins / max_distances)

        # creating bins w.r.t. distances ----
        max_distances = distances.max(dim=1)[0].view(-1, 1)  # [N x 1]
        bins, bin_sizes = self.create_bins(
            start=torch.zeros_like(max_distances), end=max_distances, steps=num_bins
        )  # [N x B]

        # creating kernels w.r.t. distances and bins ----
        bins = bins.unsqueeze(2)  # [N x B x 1]
        bins = bins.expand(-1, -1, distances.shape[1])  # [N x B x T]

        distances = distances.unsqueeze(2)  # [N x T x 1]
        distances = distances.expand(-1, -1, bins.shape[1])  # [N x T x B]
        distances = distances.permute(0, 2, 1)  # [N x B x T]

        bin_sizes = bin_sizes.unsqueeze(2)
        bin_sizes = bin_sizes.expand(-1, bins.shape[1], bins.shape[2])  # [N x B x T]

        kernels = self.sampling_kernel(bins, distances, bin_sizes)  # [N x B xT]
        max_active_bin = num_bins
        if self.drop_zero_bins:
            active_bins = kernels.sum(dim=2)  # [N x B]
            max_active_bin = 0
            # fixme: Check the full vectorized no loop way
            for n in range(N):
                i = active_bins[n].nonzero().view(-1)
                i_prime = torch.arange(i.shape[0])
                kernels[n][i_prime] = kernels[n][i]
                kernels[n][i_prime[-1] + 1 :] = 0.0
                if max_active_bin < i.shape[0]:
                    max_active_bin = i.shape[0]

            kernels = kernels[:, 0:max_active_bin]  # [N x B' x T]

        # sampling from input w.r.t. kernels ----
        input = input.permute([0, 2, 1, 3, 4])  # [N x T x C x H x W]
        input = input.flatten(start_dim=2)  # [N x T x K]
        output = torch.bmm(kernels, input)  # [N x B' x K]

        mask = output.sum(2) != 0
        output = output.view(N, max_active_bin, C, H, W).transpose(
            1, 2
        )  # [N x C x T x H x W]

        return output, mask


if __name__ == "__main__":
    features = torch.ones((1, 4, 2, 3, 3), requires_grad=True)
    sampling_params = torch.ones((1, 3, 2), requires_grad=True)
    tsl = TemporalSampling(distance_type="L1", kernel_type="linear_normalized", grad_scale_val=2.0)
    b, _ = tsl(input=features, sampling_params=sampling_params)
    loss = (b - torch.ones_like(b)).mean()
    tsl.zero_grad()
    loss.backward()
    pass
