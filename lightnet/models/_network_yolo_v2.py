#
#   Darknet YOLOv2 model
#   Copyright EAVISE
#

import functools
import logging
from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['YoloV2', 'Yolo']
log = logging.getLogger('lightnet.models')


class YoloV2(lnn.module.Darknet):
    """ Yolo v2 implementation :cite:`yolo_v2`.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (list, optional): 2D list with anchor values; Default **Yolo v2 anchors (VOC)**

    Attributes:
        self.stride: Subsampling factor of the network (input_dim / output_dim)
        self.inner_stride: Maximal internal subsampling factor of the network (input dimension should be a multiple of this)
        self.remap_darknet19: Remapping rules for weights from the :class:`~lightnet.models.Darknet19` model.
    """
    stride = 32
    inner_stride = 32
    remap_darknet19 = [
        (r'^layers.0.([1-9]_)',     r'layers.0.\1'),    # layers 1-9
        (r'^layers.0.(1[0-7]_)',    r'layers.0.\1'),    # layers 10-17
        (r'^layers.0.(\d{2}_)',     r'layers.1.\1'),    # remaining layers (18-23)
    ]

    def __init__(self, num_classes, input_channels=3, anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]):
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
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ('14_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu, momentum=momentum)),
                ('15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ('16_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu, momentum=momentum)),
                ('17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
            ]),

            # Sequence 1 : input = sequence0
            OrderedDict([
                ('18_max',          nn.MaxPool2d(2, 2)),
                ('19_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ('20_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0, relu=relu, momentum=momentum)),
                ('21_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ('22_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0, relu=relu, momentum=momentum)),
                ('23_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ('24_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ('25_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
            ]),

            # Sequence 2 : input = sequence0
            OrderedDict([
                ('26_convbatch',    lnn.layer.Conv2dBatchReLU(512, 64, 1, 1, 0, relu=relu, momentum=momentum)),
                ('27_reorg',        lnn.layer.Reorg(2)),
            ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('28_convbatch',    lnn.layer.Conv2dBatchReLU((4*64)+1024, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ('29_conv',         nn.Conv2d(1024, len(self.anchors)*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        out0 = self.layers[0](x)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out0)
        out = self.layers[3](torch.cat((out2, out1), 1))

        return out


def Yolo(*args, **kwargs):
    """ DEPRECATED : Please use the more aptly named YoloV2 class. """
    log.deprecated('This class is deprecated! Please use the more aptly named YoloV2 class.')
    return YoloV2(*args, **kwargs)
