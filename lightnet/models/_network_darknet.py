#
#   Darknet Darknet model
#   Copyright EAVISE
#

import functools
from collections import OrderedDict
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['Darknet']


class Darknet(lnn.module.Darknet):
    """ Darknet reference implementation :cite:`yolo_v1`.

    Args:
        num_classes (Number, optional): Number of classes; Default **1000**
        input_channels (Number, optional): Number of input channels; Default **3**

    Attributes:
        self.inner_stride: Maximal internal subsampling factor of the network (input dimension should be a multiple of this)
    """
    inner_stride = 64

    def __init__(self, num_classes, input_channels=3):
        super().__init__()

        # Parameters
        self.num_classes = num_classes
        self.input_channels = input_channels

        # Network
        relu = functools.partial(nn.LeakyReLU, 0.1, inplace=True)
        momentum = 0.01
        self.layers = nn.Sequential(
            # Base layers
            nn.Sequential(OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 16, 3, 1, 1, relu=relu, momentum=momentum)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchReLU(16, 32, 3, 1, 1, relu=relu, momentum=momentum)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1, relu=relu, momentum=momentum)),
                ('6_max',           nn.MaxPool2d(2, 2)),
                ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1, relu=relu, momentum=momentum)),
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1, relu=relu, momentum=momentum)),
                ('10_max',          nn.MaxPool2d(2, 2)),
                ('11_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
            ])),

            # Classification specific layers
            nn.Sequential(OrderedDict([
                ('14_avgpool',      nn.AdaptiveAvgPool2d(1)),
                ('15_conv',         nn.Conv2d(1024, num_classes, 1, 1, 0)),
                ('16_flatten',      lnn.layer.Flatten())
            ])),
        )
