#
#   Darknet YOLOv2 model with Mobilenet backend
#   Copyright EAVISE
#

import os
from collections import OrderedDict
import torch
import torch.nn as nn

import lightnet.network as lnn
import lightnet.data as lnd

__all__ = ['MobileNetYolo']


class MobileNetYolo(lnn.Darknet):
    """ `Yolo v2`_ implementation with pytorch with a `mobilenets`_ backend.
    This network uses :class:`~lightnet.network.RegionLoss` as its loss function
    and :class:`~lightnet.data.GetBoundingBoxes` as its default postprocessing function.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        weights_file (str, optional): Path to the saved weights; Default **None**
        conf_thresh (Number, optional): Confidence threshold for postprocessing of the boxes; Default **0.25**
        nms_thresh (Number, optional): Non-maxima suppression threshold for postprocessing; Default **0.4**
        alpha (Number, optional): Number between [0-1] that controls the number of filters of the mobilenet convolutions; Default **1**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (dict, optional): Dictionary containing `num` and `values` properties with anchor values; Default **Yolo v2 anchors**

    Attributes:
        self.loss (fn): loss function. Usually this is :class:`~lightnet.network.RegionLoss`
        self.postprocess (fn): Postprocessing function. By default this is :class:`~lightnet.data.GetBoundingBoxes`

    Warning:
        When changing the ``alpha`` value, you are changing the network architecture.
        This means you cannot use weights from this network with a different alpha value.
    """
    def __init__(self, num_classes=20, weights_file=None, conf_thresh=.25, nms_thresh=.4, alpha=1.0, input_channels=3, anchors=dict(num=5, values=[1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52])):
        """ Network initialisation """
        super(MobileNetYolo, self).__init__()

        # Parameters
        self.num_classes = num_classes
        self.num_anchors = anchors['num']
        self.anchors = anchors['values']
        self.reduction = 32             # input_dim/output_dim

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
                ('14_convbatch',    lnn.layer.Conv2dBatchLeaky(int(alpha*512), 64, 1, 1, 0)),
                ('15_reorg',        lnn.layer.Reorg(2)),
            ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('16_convbatch',    lnn.layer.Conv2dBatchLeaky((4*64)+int(alpha*1024), 1024, 3, 1, 1)),
                ('17_conv',         nn.Conv2d(1024, self.num_anchors*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

        self.load_weights(weights_file)
        self.loss = lnn.RegionLoss(self) 
        self.postprocess = lnd.GetBoundingBoxes(self, conf_thresh, nms_thresh)

    def _forward(self, x):
        outputs = []
    
        outputs.append(self.layers[0](x))
        outputs.append(self.layers[1](outputs[0]))
        outputs.append(self.layers[2](outputs[0]))
        out = self.layers[3](torch.cat((outputs[2], outputs[1]), 1))

        return out
