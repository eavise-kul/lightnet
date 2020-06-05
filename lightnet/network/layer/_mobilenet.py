#
#   Mobilenet related layers
#   Copyright EAVISE
#

import logging
import torch.nn as nn


__all__ = ['Conv2dDepthWise', 'InvertedBottleneck']
log = logging.getLogger(__name__)


class Conv2dDepthWise(nn.Module):
    """ This layer implements the depthwise separable convolution :cite:`mobilenet_v1`.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        momentum (int, optional): momentum of the moving averages of the normalization; Default **0.1**
        relu (class, optional): Which ReLU to use; Default :class:`torch.nn.ReLU`

    Note:
        If you require the `relu` class to get extra parameters, you can use a `lambda` or `functools.partial`:

        >>> conv = ln.layer.Conv2dDepthWise(
        ...     in_c, out_c, kernel, stride, padding,
        ...     relu=functools.partial(torch.nn.ReLU6, inplace=True)
        ... )   # doctest: +SKIP
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 momentum=0.1, relu=lambda: nn.ReLU(inplace=True)):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels, momentum=momentum),
            relu(),

            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            relu(),
        )

    def __repr__(self):
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, {relu})'
        return s.format(
            name=self.__class__.__name__,
            in_channels=self.layers[0].in_channels,
            out_channels=self.layers[3].out_channels,
            kernel_size=self.layers[0].kernel_size,
            stride=self.layers[0].stride,
            padding=self.layers[0].padding,
            relu=self.layers[2],
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class InvertedBottleneck(nn.Module):
    """ This is an implementation of the inverted bottleneck layer :cite:`mobilenet_v2`.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        expansion (int): Expansion factor for the number of channels in the depthwise convolution
        momentum (int, optional): momentum of the moving averages of the normalization; Default **0.1**
        relu (class, optional): Which ReLU to use; Default :class:`torch.nn.ReLU`

    Note:
        This layer uses a residual connection for easier propagation of the gradient.
        Whether or not this residual connection is made, depends on the ``in_channels``, ``out_channels`` and ``stride`` arguments.
        If the input channels and output channels are equal and the stride is equal to 1, the residual connection is made.

    Note:
        In order to pass arguments to your relu argument, you can use a lambda or partial application.

        >>> conv = InvertedBottleneck(
        ...     in_c, out_c, kernel, stride, expansion,
        ...     relu=functools.partial(torch.nn.ReLU6, inplace=True)
        ... )  # doctest: +SKIP
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion,
                 momentum=0.1, relu=lambda: nn.ReLU(inplace=True)):
        super().__init__()

        # Parameters
        self.expansion = expansion
        self.residual_connect = stride == 1 and in_channels == out_channels

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels*expansion, momentum=momentum),
            relu(),

            nn.Conv2d(in_channels*expansion, in_channels*expansion, kernel_size, stride, 1, groups=in_channels*expansion, bias=False),
            nn.BatchNorm2d(in_channels*expansion, momentum=momentum),
            relu(),

            nn.Conv2d(in_channels*expansion, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
        )

    def __repr__(self):
        residual = ', residual_connection' if self.residual_connect else ''
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, expansion={expansion}, {relu}{residual})'
        return s.format(
            name=self.__class__.__name__,
            in_channels=self.layers[0].in_channels,
            out_channels=self.layers[6].out_channels,
            kernel_size=self.layers[3].kernel_size,
            stride=self.layers[3].stride,
            expansion=self.expansion,
            relu=self.layers[2],
            residual=residual,
        )

    def forward(self, x):
        if self.residual_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)
