#
#   Darknet specific layers
#   Copyright EAVISE
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from .logger import *

__all__ = ['MaxPoolStride1', 'Reorg', 'GlobalAvgPool2d', 'Conv2dBatchLeaky']


class MaxPoolStride1(nn.Module):
    """ Maxpool layer with replicating padding for stride 1 """
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
    """ Reorganize a tensor according to a stride """
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
    """ Average entire channel to single number """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


class Conv2dBatchLeaky(nn.Module):
    """ Convolution layer followed by a batchnorm and a leaky ReLU """
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
