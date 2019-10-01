#
#   Mobilenet related layers
#   Copyright EAVISE
#

import logging
import torch.nn as nn


__all__ = ['Conv2dDepthWise', 'Bottleneck']
log = logging.getLogger(__name__)


class Conv2dDepthWise(nn.Module):
    """ This layer implements the depthwise separable convolution :cite:`mobilenet_v1`.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        momentum (int, optional): momentum of the moving averages of the normalization; Default **0.01**
        relu (class, optional): Which ReLU to use; Default :class:`torch.nn.ReLU6`

    Note:
        If you require the `relu` class to get extra parameters, you can use a `lambda` or `functools.partial`:

        >>> conv = ln.layer.Conv2dDepthWise(
        ...     in_c, out_c, kernel, stride, padding,
        ...     relu=functools.partial(torch.nn.ReLU6, inplace=True)
        ... )   # doctest: +SKIP
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 momentum=0.01, relu=lambda: nn.ReLU6(inplace=True)):
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
            nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels, momentum=self.momentum),
            relu(),

            nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels, momentum=self.momentum),
            relu(),
        )

    def __repr__(self):
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, {relu})'
        return s.format(name=self.__class__.__name__, relu=self.layers[2], **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class Bottleneck(nn.Module):
    """ This is an implementation of the bottleneck layer :cite:`mobilenet_v2`.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        expansion (int): Expansion factor for the number of channels in the depthwise convolution
        momentum (int, optional): momentum of the moving averages of the normalization; Default **0.01**
        relu (class, optional): Which ReLU to use; Default :class:`torch.nn.ReLU6`

    Note:
        The bottleneck layers of mobilenetv2_ use a residual connection for easier propagation of the gradient.
        Whether or not this residual connection is made, depends on the ``in_channels``, ``out_channels`` and ``stride`` arguments.
        If the input channels and output channels are equal and the stride is equal to 1, the residual connection is made.

    Note:
        The relu argument gets called with *inplace=True*.
        To give it other arguments you can use a lambda:

        >>> conv = Conv2dDepthWise(
        ...     in_c, out_c, kernel, stride, padding,
        ...     relu=functools.partial(torch.nn.ReLU6, inplace=True)
        ... )  # doctest: +SKIP
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion,
                 momentum=0.01, relu=lambda: nn.ReLU6(inplace=True)):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion = expansion
        self.momentum = momentum
        self.residual_connect = self.stride == 1 and self.in_channels == self.out_channels

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels*self.expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.in_channels*self.expansion, momentum=self.momentum),
            relu(),

            nn.Conv2d(self.in_channels*self.expansion, self.in_channels*self.expansion, self.kernel_size, self.stride, 1, groups=self.in_channels*self.expansion, bias=False),
            nn.BatchNorm2d(self.in_channels*self.expansion, momentum=self.momentum),
            relu(),

            nn.Conv2d(self.in_channels*self.expansion, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels, momentum=self.momentum),
        )

    def __repr__(self):
        residual = ', residual_connection' if self.residual_connect else ''
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, expansion={expansion}, {relu}{residual})'
        return s.format(name=self.__class__.__name__, relu=self.layers[2], residual=residual, **self.__dict__)

    def forward(self, x):
        if self.residual_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)
