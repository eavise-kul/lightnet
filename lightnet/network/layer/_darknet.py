#
#   Darknet related layers
#   Copyright EAVISE
#

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Conv2dBatchReLU', 'Flatten', 'PaddedMaxPool2d', 'Reorg']
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
        momentum (number, optional): momentum of the moving averages of the normalization; Default **0.1**
        relu (class, optional): Which ReLU to use; Default :class:`torch.nn.ReLU`

    .. figure:: /.static/api/conv2dbatchrelu.*
       :width: 100%
       :alt: Conv2dBatchReLU module design

    Note:
        The bias term in the :class:`~torch.nn.Conv2d` is disabled for this module.

    Example:
        >>> module = ln.network.layer.Conv2dBatchReLU(3, 32, 3, 1, 1)
        >>> print(module)
        Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
        >>> in_tensor = torch.rand(1, 3, 10, 10)
        >>> out_tensor = module(in_tensor)
        >>> out_tensor.shape
        torch.Size([1, 32, 10, 10])
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, momentum=0.1, relu=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            relu()
        )

    def __repr__(self):
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, {relu})'
        return s.format(
            name=self.__class__.__name__,
            in_channels=self.layers[0].in_channels,
            out_channels=self.layers[0].out_channels,
            kernel_size=self.layers[0].kernel_size,
            stride=self.layers[0].stride,
            padding=self.layers[0].padding,
            relu=self.layers[2],
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Flatten(nn.Module):
    """ Flatten tensor into single dimension.

    Args:
        batch (boolean, optional): If True, consider input to be batched and do not flatten first dim; Default **True**

    Example:
        >>> # By default batch_mode is true
        >>> module = ln.network.layer.Flatten()
        >>> in_tensor = torch.rand(8, 3, 10, 10)
        >>> out_tensor = module(in_tensor)
        >>> out_tensor.shape
        torch.Size([8, 300])

        >>> # Disable batch_mode
        >>> module = ln.network.layer.Flatten(False)
        >>> in_tensor = torch.rand(8, 3, 10, 10)
        >>> out_tensor = module(in_tensor)
        >>> out_tensor.shape
        torch.Size([2400])
    """
    def __init__(self, batch=True):
        super().__init__()
        self.batch = batch

    def forward(self, x):
        if self.batch:
            return x.view(x.size(0), -1)
        else:
            return x.view(-1)


class PaddedMaxPool2d(nn.Module):
    """ Maxpool layer with replicate-padding instead of the zero-padding from :class:`torch.nn.MaxPool2d`. |br|
    This layer is not a traditional pooling layer in the sence that it does not modify the dimension of the input tensor.

    Args:
        kernel_size (int or tuple): Kernel size for maxpooling
        stride (int or tuple, optional): The stride of the window; Default ``kernel_size``
        padding (tuple, optional): (left, right, top, bottom) padding; Default **None**
        dilation (int or tuple, optional): A parameter that controls the stride of elements in the window

    Example:
        >>> module = ln.network.layer.PaddedMaxPool2d(2, 1, (0, 1, 0, 1))
        >>> print(module)
        PaddedMaxPool2d(kernel_size=2, stride=1, padding=(0, 1, 0, 1), dilation=1)
        >>> in_tensor = torch.rand(1, 3, 10, 10)
        >>> out_tensor = module(in_tensor)
        >>> out_tensor.shape
        torch.Size([1, 3, 10, 10])
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
    The width and height dimensions (2 and 3) will be sliced by the stride and then stacked in dimension 1. (input must have 4 dimensions)

    Args:
        stride (int): stride to divide the input tensor

    Note:
        This implementation follows the darknet reorg layer implementation, which we took from this `issue <reorglink_>`_. |br|
        This specific implementation requires that the channel dimension should be divisible by :math:`stride^{\,2}`.

    Example:
        >>> # Divide width and height by 2 and stack in channels (thus multiplying by 4)
        >>> module = ln.network.layer.Reorg(stride=2)
        >>> in_tensor = torch.rand(8, 4, 10, 10)
        >>> out_tensor = module(in_tensor)
        >>> out_tensor.shape
        torch.Size([8, 16, 5, 5])

        >>> # Divide width and height by 4, Note that channel should thus be divisible by 4^2
        >>> module = ln.network.layer.Reorg(stride=4)
        >>> in_tensor = torch.rand(8, 16, 16, 16)
        >>> out_tensor = module(in_tensor)
        >>> out_tensor.shape
        torch.Size([8, 256, 4, 4])

    .. _reorglink: https://github.com/thtrieu/darkflow/issues/173#issuecomment-296048648
    """
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
        if not isinstance(stride, int):
            raise TypeError(f'stride is not an int [{type(stride)}]')

    def extra_repr(self):
        return f'stride={self.stride}'

    def forward(self, x):
        assert x.dim() == 4
        B, C, H, W = x.size()
        assert H % self.stride == 0, f'Dimension height mismatch: {H} is not divisible by {self.stride}'
        assert W % self.stride == 0, f'Dimension width mismatch: {W} is not divisible by {self.stride}'
        mem_fmt = x.is_contiguous(memory_format=torch.channels_last)

        x = x.reshape(B, C//(self.stride**2), H, self.stride, W, self.stride)
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(B, -1, H//self.stride, W//self.stride)
        if mem_fmt:
            x = x.contiguous(memory_format=torch.channels_last)

        return x
