import torch
from torch import Tensor
from torch.nn.functional import pairwise_distance, cosine_similarity


def mean_over_frames_sampling_params(func):
    """
    Distance Functions Decorator
    :return:
    """

    def mean_wrapper(features: Tensor) -> Tensor:
        """
        Calculates mean over frames features
        :param features: [N x D x T]
        :return: [N x T]
        """
        return func(features.mean(dim=2, keepdim=True), features)

    return mean_wrapper


def cov(m: Tensor, ddof: int = 1) -> Tensor:
    """
    Computes the covariance of the given batch
    :param m: [N x D x T]
    :param ddof: delta degree of freedom in d-ddof
    :return: covariance matrix of m with size [N x D X D]
    """
    n, d, t = m.shape
    if t < ddof:
        raise ValueError("number of observations should greater than ddof")
    fact = 1.0 / (t - ddof)
    mu = torch.mean(m, dim=2, keepdim=True)  # [N x D x T]
    m = m - mu  # [N x D x T]
    mt = m.transpose(1, 2)  # [N x T x D]
    return fact * torch.bmm(m, mt)


def precision_matrix(m: Tensor) -> Tensor:
    """
        Returns the precision matrix (matrix inverse of covariance matrix)
    :param m: [N x D x D]
    :return: [N x D x D]
    """
    return torch.inverse(cov(m))


# noinspection PyPep8Naming
def LP(p: int = 2):
    """
    LP Norm -> wrapping torch.nn.functional.pairwise_distance function
    :param p: p value of LP (p-norm)
    :return: pointer to the wrapper over torch.dist
    """
    # noinspection PyPep8Naming
    @mean_over_frames_sampling_params
    def wrapper(mean: Tensor, B: Tensor) -> Tensor:
        """
        wrapper over torch.nn.functional.pairwise_distance
        :param mean: mean: [N x D]
        :param B: B: [N x D x T]
        :return: torch.dist(A, B, p): [N x T] p-norm of B and A
        """
        return pairwise_distance(x1=mean, x2=B, p=p)

    return wrapper


# noinspection PyPep8Naming
@mean_over_frames_sampling_params
def Mahalanobis(mean: Tensor, input: Tensor) -> Tensor:
    """
    Squared mahalanobis distance
    :param mean: [N x D]
    :param input: [N x D x T]
    :return: Tensor: [N x T] mahalanobis of mean and input

    """

    """
    Warning
        the covariance of m can singular instead  pseudo-inverse
        (like torch.pinverse) can be used however the derivatives are not always existent.
        more info: https://pytorch.org/docs/stable/torch.html#torch.pinverse
    """
    s_inv = precision_matrix(input)  # [N x D x D]
    diff = input - mean
    diff_t = diff.transpose(1, 2)  # [N x T x D]
    mah_dist = torch.bmm(torch.bmm(diff_t, s_inv), diff)  # [N x T x T]
    mah_dist = torch.diagonal(mah_dist, dim1=-2, dim2=-1)  # [N X T]
    return mah_dist


# noinspection PyPep8Naming
@mean_over_frames_sampling_params
def Cosine(mean: Tensor, input: Tensor):
    """
    Cosine Distance = 1 - Cosine Similarity (torch.nn.functional.cosine_similarity)
    :param mean: [N x D x 1]
    :param input: [N x D x T]
    :return: Tensor: [N x T] cosine similarity of mean and input

    """
    cos_sim = cosine_similarity(input, mean, dim=1)
    return 1.0 - cos_sim



_DISTANCE_FUNCTION_TYPES = {
    "L2": LP(p=2),
    "L1": LP(p=1),
    "Mahal": Mahalanobis,
    "Cosine": Cosine,
}

