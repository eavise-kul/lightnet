#
#   Darknet related layers
#   Copyright EAVISE
#

import logging
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Conv2dBatchReLU', 'Flatten', 'GlobalAvgPool2d', 'PaddedMaxPool2d', 'Reorg']
log = logging.getLogger(__name__)


class Conv2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        momentum (int, optional): momentum of the moving averages of the normalization; Default **0.01**
        relu (class, optional): Which ReLU to use; Default :class:`torch.nn.LeakyReLU`

    Note:
        If you require the `relu` class to get extra parameters, you can use a `lambda` or `functools.partial`:

        >>> conv = ln.layer.Conv2dBatchReLU(
        ...     in_c, out_c, kernel, stride, padding,
        ...     relu=functools.partial(torch.nn.LeakyReLU, 0.1)
        ... )   # doctest: +SKIP
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 momentum=0.01, relu=lambda: nn.LeakyReLU(0.1)):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.momentum = momentum

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels, momentum=self.momentum),
            relu()
        )

    def __repr__(self):
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, {relu})'
        return s.format(name=self.__class__.__name__, relu=self.layers[2], **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class Flatten(nn.Module):
    """ Flatten tensor into single dimension.

    Args:
        batch (boolean, optional): If True, consider input to be batched and do not flatten first dim; Default **True**
    """
    def __init__(self, batch=True):
        super().__init__()
        self.batch = batch

    def forward(self, x):
        if self.batch:
            return x.view(x.size(0), -1)
        else:
            return x.view(-1)


class GlobalAvgPool2d(nn.Module):
    """ This layer averages each channel to a single number.

    Args:
        squeeze (boolean, optional): Whether to reduce dimensions to [batch, channels]; Default **True**

    Deprecated:
        This function is deprecated in favor of :class:`torch.nn.AdaptiveAvgPool2d`. |br|
        To replicate the behaviour with `squeeze` set to **True**, append a Flatten layer afterwards:

        >>> layer = torch.nn.Sequential(
        ...     torch.nn.AdaptiveAvgPool2d(1),
        ...     ln.network.layer.Flatten()
        ... )
    """
    def __init__(self, squeeze=True):
        super().__init__()
        self.squeeze = squeeze
        log.deprecated('This layer is deprecated and will be removed in future version, please use "torch.nn.AdaptiveAvgPool2d"')

    def forward(self, x):
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))

        if self.squeeze:
            x = x.view(B, C)

        return x


class PaddedMaxPool2d(nn.Module):
    """ Maxpool layer with a replicating padding.

    Args:
        kernel_size (int or tuple): Kernel size for maxpooling
        stride (int or tuple, optional): The stride of the window; Default ``kernel_size``
        padding (tuple, optional): (left, right, top, bottom) padding; Default **None**
        dilation (int or tuple, optional): A parameter that controls the stride of elements in the window
    """
    def __init__(self, kernel_size, stride=None, padding=(0, 0, 0, 0), dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}'

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, self.padding, mode='replicate'), self.kernel_size, self.stride, 0, self.dilation)
        return x


class Reorg(nn.Module):
    """ This layer reorganizes a tensor according to a stride.
    The dimensions 2,3 will be sliced by the stride and then stacked in dimension 1. (input must have 4 dimensions)

    Args:
        stride (int): stride to divide the input tensor
    """
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
        if not isinstance(stride, int):
            raise TypeError(f'stride is not an int [{type(stride)}]')

    def extra_repr(self):
        return f'stride={self.stride}'

    def forward(self, x):
        assert(x.dim() == 4)
        B, C, H, W = x.size()

        if H % self.stride != 0:
            raise ValueError(f'Dimension mismatch: {H} is not divisible by {self.stride}')
        if W % self.stride != 0:
            raise ValueError(f'Dimension mismatch: {W} is not divisible by {self.stride}')

        # from: https://github.com/thtrieu/darkflow/issues/173#issuecomment-296048648
        x = x.view(B, C//(self.stride**2), H, self.stride, W, self.stride).contiguous()
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B, -1, H//self.stride, W//self.stride)

        return x
