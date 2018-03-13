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
from torch.autograd import Function
from .reorg._ext import reorg_layer

from ..logger import *

__all__ = ['PaddedMaxPool2d', 'Reorg', 'GlobalAvgPool2d', 'Conv2dBatchLeaky']


class PaddedMaxPool2d(nn.Module):
    """ Maxpool layer with a replicating padding.

    Args:
        kernel_size (int or tuple): Kernel size for maxpooling
        stride (int or tuple, optional): The stride of the window; Default ``kernel_size``
        padding (tuple, optional): (left, right, top, bottom) padding; Default **None**
        dilation (int or tuple, optional): A parameter that controls the stride of elements in the window 
    """
    def __init__(self, kernel_size, stride=None, padding=(0,0,0,0), dilation=1):
        super(PaddedMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation

    def __repr__(self):
        return f'{self.__class__.__name__} (kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})'

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, self.padding, mode='replicate'), self.kernel_size, self.stride, 0, self.dilation)
        return x


class ReorgFunction(Function):
    def __init__(self, stride=2):
        self.stride = stride

    def forward(self, x):
        stride = self.stride

        bsize, c, h, w = x.size()
        out_w, out_h, out_c = int(w / stride), int(h / stride), c * (stride * stride)  # noqa
        out = torch.FloatTensor(bsize, out_c, out_h, out_w)

        if x.is_cuda:
            out = out.cuda()
            reorg_layer.reorg_cuda(x, w, h, c, bsize,
                                   stride, 0, out)
        else:
            reorg_layer.reorg_cpu(x, w, h, c, bsize,
                                  stride, 0, out)

        return out

    def backward(self, grad_top):
        stride = self.stride
        bsize, c, h, w = grad_top.size()

        out_w, out_h, out_c = w * stride, h * stride, c / (stride * stride)
        grad_bottom = torch.FloatTensor(bsize, int(out_c), out_h, out_w)

        # rev_stride = 1. / stride    # reverse
        if grad_top.is_cuda:
            grad_bottom = grad_bottom.cuda()
            reorg_layer.reorg_cuda(grad_top, w, h, c, bsize,
                                   stride, 1, grad_bottom)
        else:
            reorg_layer.reorg_cpu(grad_top, w, h, c, bsize,
                                  stride, 1, grad_bottom)

        return grad_bottom


class Reorg(torch.nn.Module):

    def __init__(self, stride):
        super(Reorg, self).__init__()

        self.stride = stride

    def forward(self, x):
        x = ReorgFunction(self.stride)(x)
        return x
#class Reorg(nn.Module):
#    """ This layer reorganizes a tensor according to a stride.
#    The dimensions 2,3 will be sliced by the stride and then stacked in dimension 1. (input must have 4 dimensions)
#
#    Args:
#        stride (int or tuple): stride to divide the input tensor
#    """
#    def __init__(self, stride=2):
#        super(Reorg, self).__init__()
#        if not isinstance(stride, (tuple, int)):
#            log(Loglvl.ERROR, f'stride is not a tuple or int [{type(stride)}]', TypeError)
#        self.stride = stride
#
#    def __repr__(self):
#        return f'{self.__class__.__name__} (stride={self.stride})'
#
#    def forward(self, x):
#        assert(x.data.dim() == 4)
#        B = x.data.size(0)
#        C = x.data.size(1)
#        H = x.data.size(2)
#        W = x.data.size(3)
#
#        if isinstance(self.stride, int):
#            ws = self.stride
#            hs = self.stride
#        else:
#            ws = self.stride[0]
#            hs = self.stride[1]
#        if H % hs != 0:
#            log(Loglvl.ERROR, f'Dimension mismatch: {H} is not divisible by {hs}', ValueError)
#        if W % ws != 0:
#            log(Loglvl.ERROR, f'Dimension mismatch: {W} is not divisible by {ws}', ValueError)
#
#        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
#        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
#        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
#        x = x.view(B, hs*ws*C, H//hs, W//ws)
#        return x


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
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class Conv2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2dBatchReLU, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class Conv2dDepthWise(nn.Module):
    """ This layer implements the depthwise separable convolution from Mobilenets_.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution

    .. _Mobilenets: https://arxiv.org/pdf/1704.04861.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2dDepthWise, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x
