#
#   Extra lightnet layers
#   Copyright EAVISE
#
"""
.. Note::
   Every parameter that can get an int or tuple will behave as follows. |br|
   If a tuple of 2 ints is given, the first int is used for the height and the second for the width. |br|
   If an int is given, both the width and height are set to this value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..logger import *

__all__ = ['MaxPoolStride1', 'Reorg', 'GlobalAvgPool2d', 'Conv2dBatchLeaky']


class MaxPoolStride1(nn.Module):
    """ Maxpool layer with a replicating padding, for stride 1

    Args:
        pool_size (int or tuple): pooling size
    """
    def __init__(self, pool_size=2):
        super(MaxPoolStride1, self).__init__()
        self.pool = pool_size
        self.stride = 1

    def __repr__(self):
        return f'{self.__class__.__name__} (pool_size={self.pool}, stride={self.stride})'

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), self.pool, stride=self.stride)
        return x


class Reorg(nn.Module):
    """ This layer reorganizes a tensor according to a stride.
    The dimensions 2,3 will be sliced by the stride and then stacked in dimension 1. (input must have 4 dimensions)

    Args:
        stride (int or tuple): stride to divide the input tensor
    """
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        if not isinstance(stride, (tuple, int)):
            log(Loglvl.ERROR, f'stride is not a tuple or int [{type(stride)}]', TypeError)
        self.stride = stride

    def __repr__(self):
        return f'{self.__class__.__name__} (stride={self.stride})'

    def forward(self, x):
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)

        if isinstance(self.stride, int):
            ws = self.stride
            hs = self.stride
        else:
            ws = self.stride[0]
            hs = self.stride[1]
        if H % hs != 0:
            log(Loglvl.ERROR, f'Dimension mismatch: {H} is not divisible by {hs}', ValueError)
        if W % ws != 0:
            log(Loglvl.ERROR, f'Dimension mismatch: {W} is not divisible by {ws}', ValueError)

        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x


class GlobalAvgPool2d(nn.Module):
    """ This layer averages each channel to a single number.
    """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(B, C)
        return x


class Conv2dBatchLeaky(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leaky_slope=0.1):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.leaky_slope = leaky_slope

        # Layer
        self.layer = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layer(x)
        return x
