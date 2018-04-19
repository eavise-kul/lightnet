#
#   Darknet Darknet19 model
#   Copyright EAVISE
#

import os
from collections import OrderedDict
import torch
import torch.nn as nn

import lightnet.network as lnn
import lightnet.data as lnd

__all__ = ['Darknet19']


class Darknet19(lnn.module.Darknet):
    """ `Darknet19`_ implementation with pytorch.

    Todo:
        - Loss function: L2 (Crossentropyloss in pytorch)

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        weights_file (str, optional): Path to the saved weights; Default **None**
        input_channels (Number, optional): Number of input channels; Default **3**

    Attributes:
        self.loss (fn): loss function. Usually this is :class:`~lightnet.network.RegionLoss`
        self.postprocess (fn): Postprocessing function. By default this is :class:`~lightnet.data.GetBoundingBoxes`

    .. _Darknet19: https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    """
    def __init__(self, num_classes=20, weights_file=None, input_channels=3):
        """ Network initialisation """
        super(Darknet19, self).__init__()

        # Parameters
        self.num_classes = num_classes

        # Network
        self.layers = nn.Sequential(
            OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchLeaky(input_channels, 32, 3, 1, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchLeaky(32, 64, 3, 1, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchLeaky(64, 128, 3, 1, 1)),
                ('6_convbatch',     lnn.layer.Conv2dBatchLeaky(128, 64, 1, 1, 0)),
                ('7_convbatch',     lnn.layer.Conv2dBatchLeaky(64, 128, 3, 1, 1)),
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchLeaky(128, 256, 3, 1, 1)),
                ('10_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 128, 1, 1, 0)),
                ('11_convbatch',    lnn.layer.Conv2dBatchLeaky(128, 256, 3, 1, 1)),
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('14_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 256, 1, 1, 0)),
                ('15_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('16_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 256, 1, 1, 0)),
                ('17_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('18_max',          nn.MaxPool2d(2, 2)),
                ('19_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('20_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 512, 1, 1, 0)),
                ('21_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('22_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 512, 1, 1, 0)),
                ('23_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('24_conv',         nn.Conv2d(1024, 1000, 1, 1, 0)),
                ('avgpool',         lnn.layer.GlobalAvgPool2d())
            ])
        )

        # Post
        self.loss = None
        self.postprocess = None

        if weights_file is not None:
            self.load_weights(weights_file)

    def _forward(self, x):
        return self.layers(x)
