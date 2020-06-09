#
#   Cornernet pooling layers
#   Copyright EAVISE
#

from collections import OrderedDict
import torch.nn as nn
from ._darknet import Conv2dBatchReLU
from ._util import ParallelSum

__all__ = ['TopPool', 'BottomPool', 'LeftPool', 'RightPool', 'CornerPool']


class TopPool(nn.Module):
    """ Top pooling implementation :cite:`cornernet`. """
    def forward(self, x):
        return x.flip(2).cummax(2)[0].flip(2)


class BottomPool(nn.Module):
    """ Bottom pooling implementation :cite:`cornernet`. """
    def forward(self, x):
        return x.cummax(2)[0]


class LeftPool(nn.Module):
    """ Left pooling implementation :cite:`cornernet`. """
    def forward(self, x):
        return x.flip(3).cummax(3)[0].flip(3)


class RightPool(nn.Module):
    """ Right pooling implementation :cite:`cornernet`. """
    def forward(self, x):
        return x.cummax(3)[0]


class CornerPool(nn.Module):
    """ Cornerpooling module implementation :cite:`cornernet`.

    Args:
        channels (int): Number of input and output channels
        pool1 (nn.Module): First pooling module
        pool2 (nn.Module): Second pooling module
        inter_channels (int, optional): Intermediate channels; Default **128**
        momentum (number, optional): momentum of the moving averages of the normalization; Default **0.1**
        relu (class, optional): Which ReLU to use; Default :class:`torch.nn.ReLU`

    Note:
        Compared to the `official CornerNet implementation <cornernetImpl_>`_,
        this version of cornerpooling does not add the last 3x3 Conv2dBatchReLU module.

    .. _cornernetImpl: https://github.com/princeton-vl/CornerNet-Lite/blob/6a54505d830a9d6afe26e99f0864b5d06d0bbbaf/core/models/py_utils/utils.py#L187
    """
    def __init__(self, channels, pool1, pool2, inter_channels=128, momentum=0.1, relu=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        self.layers = ParallelSum(OrderedDict([
            ('pool', ParallelSum(
                nn.Sequential(
                    Conv2dBatchReLU(channels, inter_channels, 3, 1, 1, momentum, relu),
                    pool1()
                ),
                nn.Sequential(
                    Conv2dBatchReLU(channels, inter_channels, 3, 1, 1, momentum, relu),
                    pool2()
                ),
                post=nn.Sequential(
                    nn.Conv2d(inter_channels, channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(channels),
                )
            )),
            ('conv', nn.Sequential(
                nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channels),
            )),
            ('post', relu())
        ]))

    def forward(self, x):
        return self.layers(x)
