#
#   DYolo model
#   Copyright EAVISE
#

import functools
from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['DYolo']


class DYolo(lnn.module.Lightnet):
    """ Deconvolutional Yolo (DYolo) object detector :cite:`dyolo`. |br|
    This detector is optimized for detecting small objects, by adding feature pyramids to Yolo V2.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (list, optional): 2D list with anchor values; Default **Yolo v2 anchors**

    Attributes:
        self.stride: Subsampling factor of the network (input_dim / output_dim)
        self.inner_stride: Maximal internal subsampling factor of the network (input dimension should be a multiple of this)
        self.remap_darknet19: Remapping rules for weights from the :class:`~lightnet.models.Darknet19` model.

    Note:
        As the authors of the paper did not release any configuration file,
        we just multiplied the original YoloV2-VOC anchors by 4, because this network subsamples 4 times less.
    """
    stride = 8
    inner_stride = 32
    remap_darknet19 = [
        (r'^layers.0.([1-9]_)',     r'layers.0.\1'),    # layers 1-9
        (r'^layers.0.(1[01]_)',     r'layers.0.\1'),    # layers 10-11
        (r'^layers.0.(1[2-7]_)',    r'layers.1.\1'),    # layers 12-17
        (r'^layers.0.(\d{2}_)',     r'layers.2.\1'),    # remaining layers (18-23)
    ]

    def __init__(self, num_classes, input_channels=3, anchors=[(5.2884, 6.9258), (12.771, 16.03776), (20.22348, 32.39568), (37.88448, 19.36212), (44.9456, 40.0284)]):
        super().__init__()
        if not isinstance(anchors, Iterable) and not isinstance(anchors[0], Iterable):
            raise TypeError('Anchors need to be a 2D list of numbers')

        # Parameters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.anchors = anchors

        # Network
        relu = functools.partial(nn.LeakyReLU, 0.1, inplace=True)
        momentum = 0.01
        layer_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 1, 1, relu=relu, momentum=momentum)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1, relu=relu, momentum=momentum)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1, relu=relu, momentum=momentum)),
                ('6_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0, relu=relu, momentum=momentum)),
                ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1, relu=relu, momentum=momentum)),
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1, relu=relu, momentum=momentum)),
                ('10_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0, relu=relu, momentum=momentum)),
                ('11_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1, relu=relu, momentum=momentum)),
            ]),

            # Sequence 1 : input = sequence0
            OrderedDict([
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ('14_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu, momentum=momentum)),
                ('15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ('16_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu, momentum=momentum)),
                ('17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
            ]),

            # Sequence 2 : input = sequence1
            OrderedDict([
                ('18_max',          nn.MaxPool2d(2, 2)),
                ('19_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ('20_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0, relu=relu, momentum=momentum)),
                ('21_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ('22_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0, relu=relu, momentum=momentum)),
                ('23_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ('24_deconv',       nn.ConvTranspose2d(1024, 512, 2, 2))
            ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('25_convbatch',    lnn.layer.Conv2dBatchReLU(512+512, 512, 1, 1, 0, relu=relu, momentum=momentum)),
                ('26_deconv',       nn.ConvTranspose2d(512, 512, 2, 2))
            ]),

            # Sequence 4 : input = sequence3 + sequence0
            OrderedDict([
                ('27_convbatch',    lnn.layer.Conv2dBatchReLU(512+256, 512, 1, 1, 0, relu=relu, momentum=momentum)),
                ('28_conv',         nn.Conv2d(512, len(self.anchors)*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        out0 = self.layers[0](x)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out1)
        out3 = self.layers[3](torch.cat((out2, out1), 1))
        out = self.layers[4](torch.cat((out3, out0), 1))

        return out
