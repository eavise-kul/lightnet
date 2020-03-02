#
#   Mobile YOLOv2 model
#   Copyright EAVISE
#

import functools
from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['MobileYoloV2']


class MobileYoloV2(lnn.module.Darknet):
    """ Yolo v2 implementation with depthwise separable convolutions.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (list, optional): 2D list with anchor values; Default **Yolo v2 anchors**

    Attributes:
        self.stride: Subsampling factor of the network (input dimensions should be a multiple of this number)
        self.remap_mobile_darknet19: Remapping rules for weights from the :class:`~lightnet.models.MobileDarknet19` model.
    """
    stride = 32
    remap_mobile_darknet19 = [
        (r'^layers.0.([1-9]_)',     r'layers.0.\1'),    # layers 1-9
        (r'^layers.0.(1[0-3]_)',    r'layers.0.\1'),    # layers 10-13
        (r'^layers.0.(\d{2}_)',     r'layers.1.\1'),    # remaining layers (14-19)
    ]

    def __init__(self, num_classes=20, input_channels=3, anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]):
        super().__init__()
        if not isinstance(anchors, Iterable) and not isinstance(anchors[0], Iterable):
            raise TypeError('Anchors need to be a 2D list of numbers')

        # Parameters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.anchors = anchors

        # Network
        relu = functools.partial(nn.LeakyReLU, 0.1, inplace=True)
        layer_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 2, 1, relu=relu)),
                ('2_convdw',        lnn.layer.Conv2dDepthWise(32, 64, 3, 2, 1, relu=relu)),
                ('3_convdw',        lnn.layer.Conv2dDepthWise(64, 128, 3, 1, 1, relu=relu)),
                ('4_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0, relu=relu)),
                ('5_convdw',        lnn.layer.Conv2dDepthWise(64, 128, 3, 2, 1, relu=relu)),
                ('6_convdw',        lnn.layer.Conv2dDepthWise(128, 256, 3, 1, 1, relu=relu)),
                ('7_convbatch',     lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0, relu=relu)),
                ('8_convdw',        lnn.layer.Conv2dDepthWise(128, 256, 3, 2, 1, relu=relu)),
                ('9_convdw',        lnn.layer.Conv2dDepthWise(256, 512, 3, 1, 1, relu=relu)),
                ('10_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu)),
                ('11_convdw',       lnn.layer.Conv2dDepthWise(256, 512, 3, 1, 1, relu=relu)),
                ('12_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu)),
                ('13_convdw',       lnn.layer.Conv2dDepthWise(256, 512, 3, 1, 1, relu=relu)),
            ]),

            # Sequence 1 : input = sequence0
            OrderedDict([
                ('14_max',          nn.MaxPool2d(2, 2)),
                ('15_convdw',       lnn.layer.Conv2dDepthWise(512, 1024, 3, 1, 1, relu=relu)),
                ('16_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0, relu=relu)),
                ('17_convdw',       lnn.layer.Conv2dDepthWise(512, 1024, 3, 1, 1, relu=relu)),
                ('18_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0, relu=relu)),
                ('19_convdw',       lnn.layer.Conv2dDepthWise(512, 1024, 3, 1, 1, relu=relu)),
                ('20_convdw',       lnn.layer.Conv2dDepthWise(1024, 1024, 3, 1, 1, relu=relu)),
                ('21_convdw',       lnn.layer.Conv2dDepthWise(1024, 1024, 3, 1, 1, relu=relu)),
            ]),

            # Sequence 2 : input = sequence0
            OrderedDict([
                ('22_convbatch',    lnn.layer.Conv2dBatchReLU(512, 64, 1, 1, 0, relu=relu)),
                ('23_reorg',        lnn.layer.Reorg(2)),
            ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('24_convbatch',    lnn.layer.Conv2dBatchReLU((4*64)+1024, 1024, 3, 1, 1, relu=relu)),
                ('25_conv',         nn.Conv2d(1024, len(self.anchors)*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        out0 = self.layers[0](x)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out0)
        out = self.layers[3](torch.cat((out2, out1), 1))

        return out
