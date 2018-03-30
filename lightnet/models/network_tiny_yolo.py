#
#   Darknet Tiny YOLOv2 model
#   Copyright EAVISE
#

import os
from collections import OrderedDict
import torch
import torch.nn as nn

import lightnet.network as lnn
import lightnet.data as lnd

__all__ = ['TinyYolo']


class TinyYolo(lnn.Darknet):
    """ `Tiny Yolo v2`_ implementation with pytorch.
    This network uses :class:`~lightnet.network.RegionLoss` as its loss function
    and :class:`~lightnet.data.GetBoundingBoxes` as its default postprocessing function.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        weights_file (str, optional): Path to the saved weights; Default **None**
        conf_thresh (Number, optional): Confidence threshold for postprocessing of the boxes; Default **0.25**
        nms_thresh (Number, optional): Non-maxima suppression threshold for postprocessing; Default **0.4**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (dict, optional): Dictionary containing `num` and `values` properties with anchor values; Default **Tiny yolo v2 anchors**

    Attributes:
        self.loss (fn): loss function. Usually this is :class:`~lightnet.network.RegionLoss`
        self.postprocess (fn): Postprocessing function. By default this is :class:`~lightnet.data.GetBoundingBoxes`

    .. _Tiny Yolo v2: https://github.com/pjreddie/darknet/blob/777b0982322142991e1861161e68e1a01063d76f/cfg/tiny-yolo-voc.cfg
    """
    def __init__(self, num_classes=20, weights_file=None, conf_thresh=.25, nms_thresh=.4, input_channels=3,
                anchors=dict(num=5, values=[1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52])):
        """ Network initialisation """
        super(TinyYolo, self).__init__()
        if not 'num' in anchors or not 'values' in anchors:
            raise TypeError('Incorrect anchors parameter [dict(num=..., values=[...])]')

        # Parameters
        self.num_classes = num_classes
        self.anchors = anchors
        self.reduction = 32     # input_dim/output_dim

        # Network
        layer_list = OrderedDict([
            ('1_convbatch',     lnn.layer.Conv2dBatchLeaky(input_channels, 16, 3, 1, 1)),
            ('2_max',           nn.MaxPool2d(2, 2)),
            ('3_convbatch',     lnn.layer.Conv2dBatchLeaky(16, 32, 3, 1, 1)),
            ('4_max',           nn.MaxPool2d(2, 2)),
            ('5_convbatch',     lnn.layer.Conv2dBatchLeaky(32, 64, 3, 1, 1)),
            ('6_max',           nn.MaxPool2d(2, 2)),
            ('7_convbatch',     lnn.layer.Conv2dBatchLeaky(64, 128, 3, 1, 1)),
            ('8_max',           nn.MaxPool2d(2, 2)),
            ('9_convbatch',     lnn.layer.Conv2dBatchLeaky(128, 256, 3, 1, 1)),
            ('10_max',          nn.MaxPool2d(2, 2)),
            ('11_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
            ('12_max',          lnn.layer.PaddedMaxPool2d(2, 1, (0,1,0,1))),
            ('13_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
            ('14_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 1024, 3, 1, 1)),
            ('15_conv',         nn.Conv2d(1024, self.anchors['num']*(5+self.num_classes), 1, 1, 0)),
        ])
        self.layers = nn.Sequential(layer_list)

        self.load_weights(weights_file)
        self.loss = lnn.RegionLoss(self.num_classes, self.anchors, self.reduction, self.seen)
        self.postprocess = lnd.GetBoundingBoxes(self, self.num_classes, self.anchors, conf_thresh, nms_thresh)
