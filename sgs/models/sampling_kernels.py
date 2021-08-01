from typing import Any, Type

import torch
from torch import Tensor, autograd

import sgs.utils.logging as logging

logger = logging.get_logger(__name__)


_EPSILON = 1e-7


class LinearKernel(autograd.Function):
    """
    Linear Sampling Kernel: max(0, bin_size - |dt - mu_bin|)
    """

    def __init__(self, normalize_t: bool = False, grad_scale_val=1.0):
        super(LinearKernel, self).__init__()
        global _normalize, _grad_scale_val
        _normalize = normalize_t
        _grad_scale_val = grad_scale_val

    # def forward(ctx, bins, distances):
    @staticmethod
    def forward(
        ctx: Any,
        bins: Tensor,
        distances: Tensor,
        bin_sizes: Tensor,
    ) -> Tensor:
        """
        Linear Sampling Kernel: max(0, bin_size - |dt - mu_bin|)
        :param ctx: saves data for backward
        :param bins: [N x B x T]
        :param distances: [N x B x T]
        :param bin_sizes: [N x B x T]
        :return: kernels [N x B x T]
        """
        # todo: Clean the code
        # bins = kwargs['bins']
        # bin_sizes = kwargs['bin_sizes']
        # distances = kwargs['distances']

        # todo: Clean the code
        """
        bins_mean = bins.unsqueeze(2)  # [N x B x 1]
        bins_mean = bins_mean.expand(-1, -1, distances.shape[1])  # [N x B x T]
        
        distances = distances.unsqueeze(2)  # [N x T x 1]
        distances = distances.expand(-1, -1, bins_mean.shape[1])  # [N x T x B]
        distances = distances.permute(0, 2, 1)  # [N x B x T]
        """

        zero = torch.zeros(1, dtype=distances.dtype, device=distances.device)
        kernels = torch.max(
            zero, 1 + _EPSILON - torch.abs(distances - bins)
        )  # [N x B x T]

        # kernels = torch.max(
        #     zero, 1 - torch.abs(distances - bins)
        # )  # [N x B x T]
        kernels_num_active_t = 0
        if _normalize:
            # count the number of active temporal regions for each bin
            kernels_num_active_t = (kernels != 0).sum(dim=2, keepdim=True)
            # set the zeros to one to avoid division by zero
            kernels_num_active_t[kernels_num_active_t == 0] = 1
            kernels = kernels / kernels_num_active_t

        # ctx.save_for_backward(bins , bin_sizes , distances , kernels_num_active_t)
        ctx.save_for_backward(bins, torch.ones_like(bin_sizes), distances)
        # todo: Clean the code
        # print(torch.unique(torch.abs(distances - bins).gt(bin_sizes), return_counts=True))
        # print(torch.unique(torch.abs(distances - bins).gt(1), return_counts=True))
        return kernels

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        # bins, bin_sizes, distances, kernels_num_active_t = ctx.saved_tensors
        bins, bin_sizes, distances = ctx.saved_tensors
        grad_input[torch.abs(bins - distances).gt(bin_sizes)] = 0
        # todo: Clean the code
        # grad_input[bins >= distances] = 1
        if _normalize:
            # fixme: the grad is not right
            # grad_input[bins >= distances] = 1  # kernels_num_active_t
            pass
        # todo: Check with the real gradient
        grad_input[bins < distances] = -grad_input[bins < distances]
        grad_input = grad_input * _grad_scale_val
        return (
            torch.zeros_like(grad_input),
            grad_input,
            torch.zeros_like(grad_input),
        )


class DeltaKernel(autograd.Function):
    """
    Delta Function Sampling Kernel:
    """

    def __init__(self, normalize_t: bool = False, grad_scale_val: float = 1.0):
        super(DeltaKernel, self).__init__()
        global _normalize
        _normalize = normalize_t
        global _grad_scale_val
        _grad_scale_val = grad_scale_val

    @staticmethod
    def forward(
        ctx: Any,
        bins: Tensor,
        similarity_params: Tensor,
        bin_sizes: Tensor,
    ) -> Tensor:
        """
        Delta Function Sampling Kernel Forward:
        :param ctx: saves data for backward
        :param bins: [N x B x T]
        :param similarity_params: [N x B x T]
        :param bin_sizes: [N x B x T]
        :return: kernels [N x B x T]
        """
        bin_r = bin_sizes / 2
        kernels = torch.zeros_like(bins)
        kernels[torch.abs(similarity_params - bins) <= bin_r] = 1.0
        if _normalize:
            # count the number of active temporal regions for each bin
            kernels_num_active_t = (kernels != 0).sum(dim=2, keepdim=True)
            # set the zeros to one to avoid division by zero
            kernels_num_active_t[kernels_num_active_t == 0] = 1
            kernels = kernels / kernels_num_active_t

        # ctx.save_for_backward(bins , bin_sizes , similarity_params , kernels_num_active_t)
        ctx.save_for_backward(bins , bin_r , similarity_params)

        return kernels

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        grad_input = grad_output.clone()
        # bins, bin_sizes, similarity_params, kernels_num_active_t = ctx.saved_tensors
        bins, bin_r, similarity_params = ctx.saved_tensors

        grad_input[torch.abs(similarity_params - bins) > bin_r] = 0.0
        k = 1.0
        if _normalize:
            # count the number of active temporal regions for each bin
            kernels_num_active_t = (
                (torch.abs(similarity_params - bins) <= bin_r)
                .sum(dim=2, keepdim=True)
                .expand(grad_input.shape)
                .float()
            )
            kernels_num_active_t[kernels_num_active_t == 0.0] = 1.0
            k = 1.0 / kernels_num_active_t

        grad_input = k * _grad_scale_val * grad_input

        return (
            grad_input,
            grad_input,
            grad_input,
        )



def _kernel_builder(
    kernel: Type[autograd.Function], normalize: bool = False
) -> Any:
    def wrapper(**kwargs):
        return kernel(normalize_t=normalize, **kwargs).apply

    return wrapper


_SAMPLING_KERNELS_TYPES = {
    "linear": _kernel_builder(kernel=LinearKernel, normalize=False),
    "linear_normalized": _kernel_builder(kernel=LinearKernel, normalize=True),
    "delta": _kernel_builder(kernel=DeltaKernel, normalize=False),
    "delta_normalized": _kernel_builder(kernel=DeltaKernel, normalize=True),
}
