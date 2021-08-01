from torch import nn
import torch
import torch.nn.functional as F


class DoubleAttentionLayer(nn.Module):
    """
    Implementation of Double Attention Network. NIPS 2018
    """
    def __init__(self, in_channels: int, num_frames: int, reconstruct = True):
        """

        Parameters
        ----------
        in_channels
        reconstruct: `bool` whether to re-construct output to have shape (B, in_channels, L, R)
        """
        super(DoubleAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.c_m = in_channels // 4
        self.c_n = in_channels // 4
        self.reconstruct = reconstruct
        self.avg_pool = nn.AdaptiveAvgPool3d((num_frames, 1, 1))
        self.convA = nn.Conv3d(in_channels, self.c_m, kernel_size = 1)
        self.convB = nn.Conv3d(in_channels, self.c_n, kernel_size = 1)
        self.convV = nn.Conv3d(in_channels, self.c_n, kernel_size = 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv1d(self.c_m, in_channels, kernel_size = 1)

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x: `torch.Tensor` of shape (B, C, H, W)

        Returns
        -------

        """
        batch_size, c, t, h, w = x.size()
        assert c == self.in_channels, 'input channel not equal!'
        A = self.avg_pool(self.convA(x)).view(batch_size, self.c_m, t)  # (n, c_m, t, h, w)
        B = self.avg_pool(self.convB(x)).view(batch_size, self.c_n, t) # (n, c_n, t, h, w)
        V = self.avg_pool(self.convV(x)).view(batch_size, self.c_n, t) # (n, c_n, t, h, w)
        tmpA = A.view(batch_size, self.c_m, t)
        attention_maps = B.view(batch_size, self.c_n, t)
        attention_vectors = V.view(batch_size, self.c_n, t)
        attention_maps = F.softmax(attention_maps, dim = -1)  # softmax on the last dimension to create attention maps
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim = 1)  # (B, c_n, h * w) attention on c_n dimension
        tmpZ = torch.matmul(global_descriptors, attention_vectors)  # B, self.c_m, h * w
        tmpZ = tmpZ.view(batch_size, self.c_m, t)
        if self.reconstruct: tmpZ = self.conv_reconstruct(tmpZ)
        tmpZ = tmpZ.unsqueeze(3).unsqueeze(3)
        tmpZ = tmpZ.expand_as(x)
        return tmpZ * x