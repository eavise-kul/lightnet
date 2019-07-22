#
#   Darknet YOLOv2 model with Mobilenet backend
#   Copyright EAVISE
#

from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['MobileNetYolo']


class MobileNetYolo(lnn.module.Lightnet):
    """ Yolo v2 implementation with a mobilenet v1 backend :cite:`mobilenet_v1`.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        alpha (Number, optional): Number between [0-1] that controls the number of filters of the mobilenet convolutions; Default **1**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (list, optional): 2D list with anchor values; Default **Yolo v2 anchors**

    Attributes:
        self.stride: Subsampling factor of the network (input dimensions should be a multiple of this number)

    Warning:
        When changing the ``alpha`` value, you are changing the network architecture.
        This means you cannot use weights from this network with a different alpha value.
    """
    stride = 32

    def __init__(self, num_classes=20, alpha=1.0, input_channels=3, anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]):
        super().__init__()
        if not isinstance(anchors, Iterable) and not isinstance(anchors[0], Iterable):
            raise TypeError('Anchors need to be a 2D list of numbers')

        # Parameters
        self.num_classes = num_classes
        self.anchors = anchors

        # Network
        layer_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, int(alpha*32),  3, 2, 1)),
                ('2_convdw',        lnn.layer.Conv2dDepthWise(int(alpha*32),  int(alpha*64),  3, 1, 1)),
                ('3_convdw',        lnn.layer.Conv2dDepthWise(int(alpha*64),  int(alpha*128), 3, 2, 1)),
                ('4_convdw',        lnn.layer.Conv2dDepthWise(int(alpha*128), int(alpha*128), 3, 1, 1)),
                ('5_convdw',        lnn.layer.Conv2dDepthWise(int(alpha*128), int(alpha*256), 3, 2, 1)),
                ('6_convdw',        lnn.layer.Conv2dDepthWise(int(alpha*256), int(alpha*256), 3, 1, 1)),
                ('7_convdw',        lnn.layer.Conv2dDepthWise(int(alpha*256), int(alpha*512), 3, 2, 1)),
                ('8_convdw',        lnn.layer.Conv2dDepthWise(int(alpha*512), int(alpha*512), 3, 1, 1)),
                ('9_convdw',        lnn.layer.Conv2dDepthWise(int(alpha*512), int(alpha*512), 3, 1, 1)),
            ]),

            # Sequence 1 : input = sequence0
            OrderedDict([
                ('10_convdw',       lnn.layer.Conv2dDepthWise(int(alpha*512), int(alpha*512),  3, 1, 1)),
                ('11_convdw',       lnn.layer.Conv2dDepthWise(int(alpha*512), int(alpha*512),  3, 1, 1)),
                ('12_convdw',       lnn.layer.Conv2dDepthWise(int(alpha*512), int(alpha*512),  3, 1, 1)),
                ('13_convdw',       lnn.layer.Conv2dDepthWise(int(alpha*512), int(alpha*1024), 3, 2, 1)),
            ]),

            # Sequence 2 : input = sequence0
            OrderedDict([
                ('14_convbatch',    lnn.layer.Conv2dBatchReLU(int(alpha*512), 64, 1, 1, 0)),
                ('15_reorg',        lnn.layer.Reorg(2)),
            ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('16_convbatch',    lnn.layer.Conv2dBatchReLU((4*64)+int(alpha*1024), 1024, 3, 1, 1)),
                ('17_conv',         nn.Conv2d(1024, len(self.anchors)*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        out0 = self.layers[0](x)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out0)
        out = self.layers[3](torch.cat((out2, out1), 1))

        return out
