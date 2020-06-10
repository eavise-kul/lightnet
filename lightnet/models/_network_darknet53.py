#
#   Darknet Darknet53 model
#   Copyright EAVISE
#

import functools
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
        self.inner_stride: Maximal internal subsampling factor of the network (input dimension should be a multiple of this)
    """
    inner_stride = 32

    def __init__(self, num_classes, input_channels=3):
        """ Network initialisation """
        super().__init__()

        # Parameters
        self.num_classes = num_classes

        # Network
        relu = functools.partial(nn.LeakyReLU, 0.1, inplace=True)
        momentum = 0.01
        self.layers = nn.Sequential(
            OrderedDict([
                ('1_convbatch',         lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 1, 1, relu=relu, momentum=momentum)),
                ('2_convbatch',         lnn.layer.Conv2dBatchReLU(32, 64, 3, 2, 1, relu=relu, momentum=momentum)),
                ('a_residual',          lnn.layer.Residual(OrderedDict([
                    ('3_convbatch',     lnn.layer.Conv2dBatchReLU(64, 32, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('4_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('5_convbatch',         lnn.layer.Conv2dBatchReLU(64, 128, 3, 2, 1, relu=relu, momentum=momentum)),
                ('b_residual',          lnn.layer.Residual(OrderedDict([
                    ('6_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('c_residual',          lnn.layer.Residual(OrderedDict([
                    ('8_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('9_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('10_convbatch',        lnn.layer.Conv2dBatchReLU(128, 256, 3, 2, 1, relu=relu, momentum=momentum)),
                ('d_residual',          lnn.layer.Residual(OrderedDict([
                    ('11_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('12_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('e_residual',          lnn.layer.Residual(OrderedDict([
                    ('13_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('14_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('f_residual',          lnn.layer.Residual(OrderedDict([
                    ('15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('16_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('g_residual',          lnn.layer.Residual(OrderedDict([
                    ('17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('18_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('h_residual',          lnn.layer.Residual(OrderedDict([
                    ('19_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('20_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('i_residual',          lnn.layer.Residual(OrderedDict([
                    ('21_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('22_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('j_residual',          lnn.layer.Residual(OrderedDict([
                    ('23_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('24_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('k_residual',          lnn.layer.Residual(OrderedDict([
                    ('25_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('26_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('27_convbatch',        lnn.layer.Conv2dBatchReLU(256, 512, 3, 2, 1, relu=relu, momentum=momentum)),
                ('l_residual',          lnn.layer.Residual(OrderedDict([
                    ('28_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('29_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('m_residual',          lnn.layer.Residual(OrderedDict([
                    ('30_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('31_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('n_residual',          lnn.layer.Residual(OrderedDict([
                    ('32_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('33_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('o_residual',          lnn.layer.Residual(OrderedDict([
                    ('34_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('35_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('p_residual',          lnn.layer.Residual(OrderedDict([
                    ('36_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('37_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('q_residual',          lnn.layer.Residual(OrderedDict([
                    ('38_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('39_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('r_residual',          lnn.layer.Residual(OrderedDict([
                    ('40_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('41_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('s_residual',          lnn.layer.Residual(OrderedDict([
                    ('42_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('43_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('44_convbatch',        lnn.layer.Conv2dBatchReLU(512, 1024, 3, 2, 1, relu=relu, momentum=momentum)),
                ('t_residual',          lnn.layer.Residual(OrderedDict([
                    ('45_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('46_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('u_residual',          lnn.layer.Residual(OrderedDict([
                    ('47_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('48_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('v_residual',          lnn.layer.Residual(OrderedDict([
                    ('49_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('50_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('w_residual',          lnn.layer.Residual(OrderedDict([
                    ('51_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0, relu=relu, momentum=momentum)),
                    ('52_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1, relu=relu, momentum=momentum)),
                ]))),
                ('53_avgpool',          nn.AdaptiveAvgPool2d(1)),
                ('54_conv',             nn.Conv2d(1024, num_classes, 1, 1, 0)),
                ('55_flatten',          lnn.layer.Flatten()),
            ])
        )
