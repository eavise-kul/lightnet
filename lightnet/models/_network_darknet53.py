#
#   Darknet Darknet53 model
#   Copyright EAVISE
#

from collections import OrderedDict
import torch
import torch.nn as nn

import lightnet.network as lnn

__all__ = ['Darknet53']


class Darknet53(lnn.module.Darknet):
    """ Darknet53 implementation :cite:`yolo_v3`.

    Args:
        num_classes (Number, optional): Number of classes; Default **1000**
        input_channels (Number, optional): Number of input channels; Default **3**

    Attributes:
        self.stride: Subsampling factor of the network (input dimensions should be a multiple of this number)
    """
    stride = 32

    def __init__(self, num_classes=1000, input_channels=3):
        """ Network initialisation """
        super().__init__()

        # Parameters
        self.num_classes = num_classes

        # Network
        self.layers = nn.Sequential(
            OrderedDict([
                ('1_convbatch',         lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 1, 1)),
                ('2_convbatch',         lnn.layer.Conv2dBatchReLU(32, 64, 3, 2, 1)),
                ('a_residual',          lnn.layer.Residual(OrderedDict([
                    ('3_convbatch',     lnn.layer.Conv2dBatchReLU(64, 32, 1, 1, 0)),
                    ('4_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
                ]))),
                ('5_convbatch',         lnn.layer.Conv2dBatchReLU(64, 128, 3, 2, 1)),
                ('b_residual',          lnn.layer.Residual(OrderedDict([
                    ('6_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0)),
                    ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ]))),
                ('c_residual',          lnn.layer.Residual(OrderedDict([
                    ('8_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0)),
                    ('9_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ]))),
                ('10_convbatch',        lnn.layer.Conv2dBatchReLU(128, 256, 3, 2, 1)),
                ('d_residual',          lnn.layer.Residual(OrderedDict([
                    ('11_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('12_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('e_residual',          lnn.layer.Residual(OrderedDict([
                    ('13_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('14_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('f_residual',          lnn.layer.Residual(OrderedDict([
                    ('15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('16_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('g_residual',          lnn.layer.Residual(OrderedDict([
                    ('17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('18_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('h_residual',          lnn.layer.Residual(OrderedDict([
                    ('19_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('20_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('i_residual',          lnn.layer.Residual(OrderedDict([
                    ('21_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('22_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('j_residual',          lnn.layer.Residual(OrderedDict([
                    ('23_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('24_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('k_residual',          lnn.layer.Residual(OrderedDict([
                    ('25_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('26_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('27_convbatch',        lnn.layer.Conv2dBatchReLU(256, 512, 3, 2, 1)),
                ('l_residual',          lnn.layer.Residual(OrderedDict([
                    ('28_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('29_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('m_residual',          lnn.layer.Residual(OrderedDict([
                    ('30_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('31_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('n_residual',          lnn.layer.Residual(OrderedDict([
                    ('32_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('33_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('o_residual',          lnn.layer.Residual(OrderedDict([
                    ('34_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('35_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('p_residual',          lnn.layer.Residual(OrderedDict([
                    ('36_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('37_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('q_residual',          lnn.layer.Residual(OrderedDict([
                    ('38_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('39_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('r_residual',          lnn.layer.Residual(OrderedDict([
                    ('40_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('41_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('s_residual',          lnn.layer.Residual(OrderedDict([
                    ('42_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('43_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('44_convbatch',        lnn.layer.Conv2dBatchReLU(512, 1024, 3, 2, 1)),
                ('t_residual',          lnn.layer.Residual(OrderedDict([
                    ('45_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('46_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('u_residual',          lnn.layer.Residual(OrderedDict([
                    ('47_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('48_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('v_residual',          lnn.layer.Residual(OrderedDict([
                    ('49_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('50_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('w_residual',          lnn.layer.Residual(OrderedDict([
                    ('51_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('52_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('53_avgpool',          nn.AdaptiveAvgPool2d(1)),
                ('54_conv',             nn.Conv2d(1024, num_classes, 1, 1, 0)),
                ('55_flatten',          lnn.layer.Flatten()),
            ])
        )
