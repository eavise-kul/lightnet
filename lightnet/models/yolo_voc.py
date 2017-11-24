#
#   Darknet Yolo-voc model
#   By Tanguy Ophoff
#

import os
from collections import OrderedDict
import torch
import torch.nn as nn

import lightnet as ln
from .. import layers as lnl


class YoloVoc(ln.Darknet):
    """ yolo-voc.cfg implementation with pytorch """
    def __init__(self, num_classes=20, weights_file=None):
        """ Network initialisation """
        super(YoloVoc, self).__init__()

        # Parameters
        self.input_dim = (416, 416, 3)
        self.num_classes = num_classes
        self.anchors = [1.3221, 1.73145,    3.19275, 4.00944,   5.05587, 8.09892,   9.47112, 4.84053,   11.2364, 10.0071]
        self.num_anchors = 5

        # Network
        layer_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('1_convbatch',     lnl.Conv2dBatchLeaky(self.input_dim[2], 32, 3, 1, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnl.Conv2dBatchLeaky(32, 64, 3, 1, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnl.Conv2dBatchLeaky(64, 128, 3, 1, 1)),
                ('6_convbatch',     lnl.Conv2dBatchLeaky(128, 64, 1, 1, 0)),
                ('7_convbatch',     lnl.Conv2dBatchLeaky(64, 128, 3, 1, 1)),
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     lnl.Conv2dBatchLeaky(128, 256, 3, 1, 1)),
                ('10_convbatch',    lnl.Conv2dBatchLeaky(256, 128, 1, 1, 0)),
                ('11_convbatch',    lnl.Conv2dBatchLeaky(128, 256, 3, 1, 1)),
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnl.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('14_convbatch',    lnl.Conv2dBatchLeaky(512, 256, 1, 1, 0)),
                ('15_convbatch',    lnl.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('16_convbatch',    lnl.Conv2dBatchLeaky(512, 256, 1, 1, 0)),
                ('17_convbatch',    lnl.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
            ]),

            # Sequence 1 : input = sequence0
            OrderedDict([
                ('18_max',          nn.MaxPool2d(2, 2)),
                ('19_convbatch',    lnl.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('20_convbatch',    lnl.Conv2dBatchLeaky(1024, 512, 1, 1, 0)),
                ('21_convbatch',    lnl.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('22_convbatch',    lnl.Conv2dBatchLeaky(1024, 512, 1, 1, 0)),
                ('23_convbatch',    lnl.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('24_convbatch',    lnl.Conv2dBatchLeaky(1024, 1024, 3, 1, 1)),
                ('25_convbatch',    lnl.Conv2dBatchLeaky(1024, 1024, 3, 1, 1)),
            ]),

            # Sequence 2 : input = sequence0
            OrderedDict([
                ('26_convbatch',    lnl.Conv2dBatchLeaky(512, 64, 1, 1, 0)),
                ('27_reorg',        lnl.Reorg(2)),
            ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('28_convbatch',    lnl.Conv2dBatchLeaky((4*64)+1024, 1024, 3, 1, 1)),
                ('29_conv',         nn.Conv2d(1024, self.num_anchors*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

        # Weights
        self.load_weights(weights_file)

        # Loss
        self.loss = ln.RegionLoss(self.num_classes, self.anchors, self.num_anchors) 
        self.loss.seen = self.seen

        # Postprocessing
        conf_thresh = 0.25
        nms_thresh = 0.4
        self.postprocess = ln.BBoxConverter(self, conf_thresh, nms_thresh)

    def _forward(self, x):
        outputs = []
    
        outputs.append(self.layers[0](x))
        outputs.append(self.layers[1](outputs[0]))
        # Route : layers=-9
        outputs.append(self.layers[2](outputs[0]))
        # Route : layers=-1,-4
        out = self.layers[3](torch.cat((outputs[2], outputs[1]), 1))

        return out
