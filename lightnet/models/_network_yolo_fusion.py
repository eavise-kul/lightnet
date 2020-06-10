#
#   Darknet YOLOv2 with sensor fusion
#   Copyright EAVISE
#

import functools
from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['YoloFusion']


class YoloFusion(lnn.module.Lightnet):
    """ Yolo v2 network that is able to fuse multiple image sensor data at a certain parameterizable point in the network :cite:`rgbd_fusion_v2`.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        input_channels (int, optional): Number of input channels for the main subnetwork; Default **3**
        fusion_channels (int, optional): Number of input channels for the fusion subnetwork; Default **1**
        fuse_layer (int, optional): Number between 0-28, that controls at which layer to fuse both convolutional streams; Default **0**
        anchors (list, optional): 2D list with anchor values; Default **Yolo v2 anchors**

    Attributes:
        self.stride: Subsampling factor of the network (input_dim / output_dim)
        self.inner_stride: Maximal internal subsampling factor of the network (input dimension should be a multiple of this)

    Note:
        This network effectively supersedes the networks from :cite:`rgbd_fusion_v1`. |br|
        If you still want the old networks, you can take a look at a `previous version of lightnet`_.
        (Note that this is an older version of lightnet and thus some things might be broken in the latest version)

    .. _previous version of lightnet: https://gitlab.com/EAVISE/lightnet/blob/59baa61e429f63f80334dfff3ec2304d788ba1ad/lightnet/models/_network_yolo_fusion.py
    """
    stride = 32
    inner_stride = 32

    def __init__(self, num_classes, input_channels=3, fusion_channels=1, fuse_layer=0,
                 anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]):
        super().__init__()
        if not isinstance(anchors, Iterable) and not isinstance(anchors[0], Iterable):
            raise TypeError('Anchors need to be a 2D list of numbers')
        if input_channels < 1 or fusion_channels < 1:
            raise ValueError('input_channels and fusion_channels need to be at least 1 [{input_channels}, {fusion_channels}]')

        # Parameters
        self.num_classes = num_classes
        self.anchors = anchors
        self.input_channels = input_channels
        self.fusion_channels = fusion_channels
        self.fuse_layer = fuse_layer
        self.fuse_seq = None

        relu = functools.partial(nn.LeakyReLU, 0.1, inplace=True)
        momentum = 0.01

        # First layer
        if fuse_layer == 0:
            self.fuse_seq = 0
            self.layers = [
                nn.Sequential(OrderedDict([
                    ('fuse', nn.Conv2d(input_channels+fusion_channels, input_channels, 1, 1, 0, bias=False)),
                    ('1_convbatch', lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))
            ]
        elif fuse_layer == 1:
            self.fuse_seq = 0
            self.layers = [
                nn.ModuleDict({
                    '1_convbatch_regular': lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 1, 1, relu=relu, momentum=momentum),
                    '1_convbatch_fusion': lnn.layer.Conv2dBatchReLU(fusion_channels, 32, 3, 1, 1, relu=relu, momentum=momentum),
                    'fuse': nn.Conv2d(32*2, 32, 1, 1, 0, bias=False),
                })
            ]
        else:
            self.layers = [
                nn.ModuleDict({
                    '1_convbatch_regular': lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 1, 1, relu=relu, momentum=momentum),
                    '1_convbatch_fusion': lnn.layer.Conv2dBatchReLU(fusion_channels, 32, 3, 1, 1, relu=relu, momentum=momentum),
                })
            ]

        # Main layers
        layer_list = [
            OrderedDict([
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

            OrderedDict([
                ('26_convbatch',    lnn.layer.Conv2dBatchReLU((4*64)+1024, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ('27_conv',         nn.Conv2d(1024, len(self.anchors)*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        i = 1
        for e, l in enumerate(layer_list, 1):
            if fuse_layer - i <= 0:
                fuse = None
            elif fuse_layer - i <= len(l):
                fuse = fuse_layer - i
                self.fuse_seq = e
            else:
                fuse = len(l) + 1
            i += len(l)

            self.layers.append(lnn.layer.Fusion(l, fuse))

        if self.fuse_seq is None:
            raise ValueError(f'Fuse_layer too high [{fuse_layer}/{sum([len(l) for l in layer_list])+1}]')

        # Passthrough layer
        if self.fuse_seq <= 2:
            self.layers.append(nn.Sequential(
                OrderedDict([
                    ('P1_convbatch',    lnn.layer.Conv2dBatchReLU(512, 64, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('P2_reorg',        lnn.layer.Reorg(2)),
                ])
            ))
        else:
            self.layers.append(lnn.layer.Fusion(
                OrderedDict([
                    ('P1_convbatch',    lnn.layer.Conv2dBatchReLU(512, 64, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('P2_reorg',        lnn.layer.Reorg(2)),
                ]), 3
            ))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        if x.size(1) != self.input_channels + self.fusion_channels:
            raise TypeError(f'This network requires {self.input_channels+self.fusion_channels} channel input images')

        # First layer
        if self.fuse_layer <= 0:
            x = self.layers[0](x)
        else:
            r = self.layers[0]['1_convbatch_regular'](x[:, :self.input_channels])
            f = self.layers[0]['1_convbatch_fusion'](x[:, self.input_channels:])
            x = torch.cat((r, f), 1)
            if 'fuse' in self.layers[0]:
                x = self.layers[0]['fuse'](x)

        # Sequence 1
        x = self.layers[1](x)

        # Passthrough
        if self.fuse_seq == 2:
            p = self.layers[4](x[:, :x.size(1)//2])
        else:
            p = self.layers[4](x)

        # Sequence 2
        x = self.layers[2](x)

        # Sequence 3
        if self.fuse_seq == 3:
            xs = x.size(1) // 2
            ps = p.size(1) // 2
            x = torch.cat((x[:, :xs], p[:, :ps], x[:, xs:], p[:, ps:]), 1)
        else:
            x = torch.cat((x, p), 1)
        x = self.layers[3](x)

        return x
