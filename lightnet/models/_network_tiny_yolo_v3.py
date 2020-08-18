#
#   Darknet Tiny YOLOv3 model
#   Copyright EAVISE
#

import functools
from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['TinyYoloV3']


class TinyYoloV3(lnn.module.Darknet):
    """ Tiny Yolo v3 implementation :cite:`yolo_v3`.

    Args:
        num_classes (Number, optional): Number of classes; Default **80**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (list, optional): 3D list with anchor values; Default **Tiny Yolo v3 anchors (COCO)**

    Attributes:
        self.stride: Subsampling factors of the network (input_dim / output_dim)
        self.inner_stride: Maximal internal subsampling factor of the network (input dimension should be a multiple of this)
        self.remap_darknet: Remapping rules for weights from the `~lightnet.models.Darknet` model.

    Note:
        Unlike YoloV2, the anchors here are defined as multiples of the input dimensions and not as a multiple of the output dimensions!
        The anchor list also has one more dimension than the one from YoloV2, in order to differentiate which anchors belong to which stride.

    Warning:
        The :class:`~lightnet.network.loss.MultiScaleRegionLoss` and :class:`~lightnet.data.transform.GetMultiScaleBoundingBoxes`
        do not implement the overlapping class labels of the original implementation.
        Your weight files from darknet will thus not have the same accuracies as in darknet itself.
    """
    stride = (32, 16)
    inner_stride = 32
    remap_darknet = [
        (r'^layers.0.(\d+_)',   r'extractor.\1'),  # All base layers (1-13)
    ]

    def __init__(self, num_classes, input_channels=3, anchors=[[(81, 82), (135, 169), (344, 319)], [(10, 14), (23, 27), (37, 58)]]):
        super().__init__()
        if not isinstance(anchors, Iterable) and not isinstance(anchors[0], Iterable) and not isinstance(anchors[0][0], Iterable):
            raise TypeError('Anchors need to be a 3D list of numbers')

        # Parameters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.anchors = []   # YoloV3 defines anchors as a multiple of the input dimensions of the network as opposed to the output dimensions
        for i, s in enumerate(self.stride):
            self.anchors.append([(a[0] / s, a[1] / s) for a in anchors[i]])

        # Network
        relu = functools.partial(nn.LeakyReLU, 0.1, inplace=True)
        momentum = 0.01
        self.extractor = lnn.layer.SequentialSelect(
            ['9_convbatch'], True,
            # Sequence 0 : input = input_channels
            OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 16, 3, 1, 1, relu=relu, momentum=momentum)),
                ('2_maxpool',       nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchReLU(16, 32, 3, 1, 1, relu=relu, momentum=momentum)),
                ('4_maxpool',       nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1, relu=relu, momentum=momentum)),
                ('6_maxpool',       nn.MaxPool2d(2, 2)),
                ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1, relu=relu, momentum=momentum)),
                ('8_maxpool',       nn.MaxPool2d(2, 2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1, relu=relu, momentum=momentum)),
                ('10_maxpool',      nn.MaxPool2d(2, 2)),
                ('11_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ('12_maxpool',      lnn.layer.PaddedMaxPool2d(2, 1, (0, 1, 0, 1))),
                ('13_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ('14_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 256, 1, 1, 0, relu=relu, momentum=momentum)),
            ]),
        )

        self.detector = nn.ModuleList([
            # Sequence 0 : input = extractor
            nn.Sequential(
                OrderedDict([
                    ('15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                    ('16_conv',         nn.Conv2d(512, len(self.anchors[0])*(5+self.num_classes), 1, 1, 0)),
                ])
            ),

            # Sequence 1 : input = extractor
            nn.Sequential(
                OrderedDict([
                    ('17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('18_upsample',     nn.Upsample(scale_factor=2, mode='nearest')),
                ])
            ),

            # Sequence 2 : input = 18_upsample + 9_convbatch
            nn.Sequential(
                OrderedDict([
                    ('19_convbatch',    lnn.layer.Conv2dBatchReLU(128+256, 256, 3, 1, 1, relu=relu, momentum=momentum)),
                    ('20_conv',         nn.Conv2d(256, len(self.anchors[1])*(5+self.num_classes), 1, 1, 0)),
                ])
            ),
        ])

    def forward(self, x):
        # Feature extractor
        x, inter_features = self.extractor(x)

        # Detector 0
        out_0 = self.detector[0](x)

        # Detector 1
        x = self.detector[1](x)
        out_1 = self.detector[2](torch.cat((x, inter_features), 1))

        return (out_0, out_1)
