#
#   Cornernet model
#   Copyright EAVISE
#

import functools
import logging
import re
from collections import OrderedDict
import torch
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['Cornernet']
log = logging.getLogger('lightnet.models')


class Cornernet(lnn.module.Lightnet):
    """ Cornernet implementation :cite:`cornernet`.

    .. admonition:: Experimental

       This network implementation is still in development
       and might not be yielding the same results as the official implementation.
       Use at your own risk!

    Args:
        num_classes (Number): Number of classes
        input_channels (Number, optional): Number of input channels; Default **3**
        inference_only (boolean, optional): Whether to load the model purely for inference; Default **False**

    Attributes:
        self.stride: Subsampling factor of the network (input_dim / output_dim)
        self.inner_stride: Maximal internal subsampling factor of the network (input dimension should be a multiple of this)
        self.remap_princeton_vl: Remapping rules for weights from the `official CornerNet implementation <cornernetImpl_>`_.

    .. _cornernetImpl: https://github.com/princeton-vl/CornerNet-Lite
    """
    stride = 4
    inner_stride = 128

    def __init__(self, num_classes, input_channels=3, inference_only=False):
        super().__init__()
        log.experimental(f'"{self.__class__.__name__}" is still in development. Use at your own risk!')

        # Parameters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.inference_only = inference_only

        # Network
        self.extractor = nn.Sequential(OrderedDict([
            ('1_convbatch',                 lnn.layer.Conv2dBatchReLU(input_channels, 128, 7, 2, 3)),
            ('2_residual',                  self.get_residual(128, 256, stride=2)),
            ('para',                        lnn.layer.ParallelSum(
                lnn.layer.SequentialSelect(['4_convbatch'], False, OrderedDict([
                    ('3_hourglass',         self.get_hourglass(self.get_residual)),
                    ('4_convbatch',         lnn.layer.Conv2dBatchReLU(256, 256, 3, 1, 1)),
                    ('5_conv',              nn.Conv2d(256, 256, 1, 1, 0, bias=False)),
                    ('6_batchnorm',         nn.BatchNorm2d(256)),
                ])),
                nn.Sequential(OrderedDict([
                    ('7_conv',              nn.Conv2d(256, 256, 1, 1, 0, bias=False)),
                    ('8_batchnorm',         nn.BatchNorm2d(256)),
                ])),
                post=nn.ReLU(inplace=True),
            )),
            ('9_residual',                  self.get_residual(256, 256)),
            ('10_hourglass',                self.get_hourglass(self.get_residual)),
            ('11_convbatch',                lnn.layer.Conv2dBatchReLU(256, 256, 3, 1, 1))
        ]))

        self.detector = lnn.layer.ParallelCat(OrderedDict([
            ('topleft',                     nn.Sequential(OrderedDict([
                ('12_corner',               lnn.layer.CornerPool(256, lnn.layer.TopPool, lnn.layer.LeftPool)),
                ('13_convbatch',            lnn.layer.Conv2dBatchReLU(256, 256, 3, 1, 1)),
                ('output',                  lnn.layer.ParallelCat(OrderedDict([
                    ('heatmap',             nn.Sequential(OrderedDict([
                        ('14_conv',         nn.Conv2d(256, 256, 3, 1, 1, bias=True)),
                        ('15_relu',         nn.ReLU(inplace=True)),
                        ('16_conv',         nn.Conv2d(256, num_classes, 1, 1, 0)),
                    ]))),
                    ('embedding',           nn.Sequential(OrderedDict([
                        ('17_conv',         nn.Conv2d(256, 256, 3, 1, 1, bias=True)),
                        ('18_relu',         nn.ReLU(inplace=True)),
                        ('19_conv',         nn.Conv2d(256, 1, 1, 1, 0)),
                    ]))),
                    ('offset',              nn.Sequential(OrderedDict([
                        ('20_conv',         nn.Conv2d(256, 256, 3, 1, 1, bias=True)),
                        ('21_relu',         nn.ReLU(inplace=True)),
                        ('22_conv',         nn.Conv2d(256, 2, 1, 1, 0)),
                    ]))),
                ]))),
            ]))),
            ('bottomright',                 nn.Sequential(OrderedDict([
                ('23_corner',               lnn.layer.CornerPool(256, lnn.layer.BottomPool, lnn.layer.RightPool)),
                ('24_convbatch',            lnn.layer.Conv2dBatchReLU(256, 256, 3, 1, 1)),
                ('output',                  lnn.layer.ParallelCat(OrderedDict([
                    ('heatmap',             nn.Sequential(OrderedDict([
                        ('25_conv',         nn.Conv2d(256, 256, 3, 1, 1, bias=True)),
                        ('26_relu',         nn.ReLU(inplace=True)),
                        ('27_conv',         nn.Conv2d(256, num_classes, 1, 1, 0)),
                    ]))),
                    ('embedding',           nn.Sequential(OrderedDict([
                        ('28_conv',         nn.Conv2d(256, 256, 3, 1, 1, bias=True)),
                        ('29_relu',         nn.ReLU(inplace=True)),
                        ('30_conv',         nn.Conv2d(256, 1, 1, 1, 0)),
                    ]))),
                    ('offset',              nn.Sequential(OrderedDict([
                        ('31_conv',         nn.Conv2d(256, 256, 3, 1, 1, bias=True)),
                        ('32_relu',         nn.ReLU(inplace=True)),
                        ('33_conv',         nn.Conv2d(256, 2, 1, 1, 0)),
                    ]))),
                ]))),
            ]))),
        ]))

        if not self.inference_only:
            self.intermediate = lnn.layer.ParallelCat(OrderedDict([
                ('topleft',                     nn.Sequential(OrderedDict([
                    ('34_corner',               lnn.layer.CornerPool(256, lnn.layer.TopPool, lnn.layer.LeftPool)),
                    ('35_convbatch',            lnn.layer.Conv2dBatchReLU(256, 256, 3, 1, 1)),
                    ('output',                  lnn.layer.ParallelCat(OrderedDict([
                        ('heatmap',             nn.Sequential(OrderedDict([
                            ('36_conv',         nn.Conv2d(256, 256, 3, 1, 1, bias=True)),
                            ('37_relu',         nn.ReLU(inplace=True)),
                            ('38_conv',         nn.Conv2d(256, num_classes, 1, 1, 0)),
                        ]))),
                        ('embedding',           nn.Sequential(OrderedDict([
                            ('39_conv',         nn.Conv2d(256, 256, 3, 1, 1, bias=True)),
                            ('40_relu',         nn.ReLU(inplace=True)),
                            ('41_conv',         nn.Conv2d(256, 1, 1, 1, 0)),
                        ]))),
                        ('offset',              nn.Sequential(OrderedDict([
                            ('42_conv',         nn.Conv2d(256, 256, 3, 1, 1, bias=True)),
                            ('43_relu',         nn.ReLU(inplace=True)),
                            ('44_conv',         nn.Conv2d(256, 2, 1, 1, 0)),
                        ]))),
                    ]))),
                ]))),
                ('bottomright',                 nn.Sequential(OrderedDict([
                    ('45_corner',               lnn.layer.CornerPool(256, lnn.layer.BottomPool, lnn.layer.RightPool)),
                    ('46_convbatch',            lnn.layer.Conv2dBatchReLU(256, 256, 3, 1, 1)),
                    ('output',                  lnn.layer.ParallelCat(OrderedDict([
                        ('heatmap',             nn.Sequential(OrderedDict([
                            ('47_conv',         nn.Conv2d(256, 256, 3, 1, 1, bias=True)),
                            ('48_relu',         nn.ReLU(inplace=True)),
                            ('49_conv',         nn.Conv2d(256, num_classes, 1, 1, 0)),
                        ]))),
                        ('embedding',           nn.Sequential(OrderedDict([
                            ('50_conv',         nn.Conv2d(256, 256, 3, 1, 1, bias=True)),
                            ('51_relu',         nn.ReLU(inplace=True)),
                            ('52_conv',         nn.Conv2d(256, 1, 1, 1, 0)),
                        ]))),
                        ('offset',              nn.Sequential(OrderedDict([
                            ('53_conv',         nn.Conv2d(256, 256, 3, 1, 1, bias=True)),
                            ('54_relu',         nn.ReLU(inplace=True)),
                            ('55_conv',         nn.Conv2d(256, 2, 1, 1, 0)),
                        ]))),
                    ]))),
                ]))),
            ]))

        # Set mode
        if self.inference_only:
            self.eval()

    def forward(self, x):
        return self.forward_train(x) if self.training else self.forward_test(x)

    def forward_train(self, x):
        x = self.extractor(x)
        out1 = self.detector(x)
        out2 = self.intermediate(self.extractor[2][0].selected)
        return out1, out2

    def forward_test(self, x):
        x = self.extractor(x)
        return self.detector(x)

    def train(self, mode=True):
        if mode and self.inference_only:
            raise ValueError("Cannot set training mode for inference_only model")
        return super().train(mode)

    @staticmethod
    def get_residual(in_channels, out_channels, kernel=3, stride=1, padding=1):
        return lnn.layer.Residual(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels),

            skip=None if (in_channels == out_channels) and (stride == 1) else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            post=nn.ReLU(inplace=True),
        )

    @staticmethod
    def get_hourglass(residual):
        return lnn.layer.HourGlass(
            5, [256, 256, 384, 384, 384, 512],
            make_upper=lambda ci, co: nn.Sequential(*[residual(ci, co) for _ in range(2)]),
            make_down1=lambda ci, co: nn.Sequential(residual(ci, co, stride=2), residual(co, co)),
            make_inner=lambda ci, co: nn.Sequential(*[residual(ci, co) for _ in range(4)]),
            make_down2=lambda ci, co: nn.Sequential(residual(ci, ci), residual(ci, co), nn.Upsample(scale_factor=2, mode='nearest'))
        )

    @staticmethod
    def remap_princeton_vl(k):
        remap_extractor = [
            (r'^module.hg.pre.0.conv.(.*)',     r'extractor.1_convbatch.layers.0.\1'),
            (r'^module.hg.pre.0.bn.(.*)',       r'extractor.1_convbatch.layers.1.\1'),
            (r'^module.hg.pre.1.conv1.(.*)',    r'extractor.2_residual.0.\1'),
            (r'^module.hg.pre.1.bn1.(.*)',      r'extractor.2_residual.1.\1'),
            (r'^module.hg.pre.1.conv2.(.*)',    r'extractor.2_residual.3.\1'),
            (r'^module.hg.pre.1.bn2.(.*)',      r'extractor.2_residual.4.\1'),
            (r'^module.hg.pre.1.skip.(.*)',     r'extractor.2_residual.skip.\1'),
            (r'^module.hg.cnvs.0.conv.(.*)',    r'extractor.para.0.4_convbatch.layers.0.\1'),
            (r'^module.hg.cnvs.0.bn.(.*)',      r'extractor.para.0.4_convbatch.layers.1.\1'),
            (r'^module.hg.cnvs.1.conv.(.*)',    r'extractor.11_convbatch.layers.0.\1'),
            (r'^module.hg.cnvs.1.bn.(.*)',      r'extractor.11_convbatch.layers.1.\1'),
            (r'^module.hg.cnvs_.0.0.(.*)',      r'extractor.para.0.5_conv.\1'),
            (r'^module.hg.cnvs_.0.1.(.*)',      r'extractor.para.0.6_batchnorm.\1'),
            (r'^module.hg.inters.0.conv1.(.*)', r'extractor.9_residual.0.\1'),
            (r'^module.hg.inters.0.bn1.(.*)',   r'extractor.9_residual.1.\1'),
            (r'^module.hg.inters.0.conv2.(.*)', r'extractor.9_residual.3.\1'),
            (r'^module.hg.inters.0.bn2.(.*)',   r'extractor.9_residual.4.\1'),
            (r'^module.hg.inters.0.skip.(.*)',  r'extractor.9_residual.skip.\1'),
            (r'^module.hg.inters_.0.0.(.*)',    r'extractor.para.1.7_conv.\1'),
            (r'^module.hg.inters_.0.1.(.*)',    r'extractor.para.1.8_batchnorm.\1'),
        ]
        for r in remap_extractor:
            if re.match(r[0], k) is not None:
                return re.sub(r[0], r[1], k)

        # HOURGLASSES
        if k.startswith('module.hg.hgs'):
            if k[14] == '0':
                nk = 'extractor.para.0.3_hourglass.layers.'
            else:
                nk = 'extractor.10_hourglass.layers.'

            k = (
                k[16:]
                .replace('up1', 'upper')
                .replace('low1', 'down.down1')
                .replace('low3', 'down.down2')
                .replace('conv1', '0')
                .replace('bn1', '1')
                .replace('conv2', '3')
                .replace('bn2', '4')
            )
            k = re.sub(r'low2.(\d+)', r'down.inner.\1', k)
            k = k.replace('low2', 'down.inner.layers')

            return nk+k

        # DETECTION
        _, mod, num, rk = k.split('.', 3)
        corner, mod = mod.split('_')

        if num == '0':
            nk = 'intermediate.'
            num = 22
        else:
            nk = 'detector.'
            num = 0
        if corner == 'tl':
            nk += 'topleft.'
        else:
            nk += 'bottomright.'
            num += 11

        if mod == 'modules':
            remap_detection = [
                (r'^p1_conv1.conv.(.*)',    f'{12+num}_corner.layers.pool.0.0.layers.0.\\1'),
                (r'^p1_conv1.bn.(.*)',      f'{12+num}_corner.layers.pool.0.0.layers.1.\\1'),
                (r'^p2_conv1.conv.(.*)',    f'{12+num}_corner.layers.pool.1.0.layers.0.\\1'),
                (r'^p2_conv1.bn.(.*)',      f'{12+num}_corner.layers.pool.1.0.layers.1.\\1'),
                (r'^p_conv1.(.*)',          f'{12+num}_corner.layers.pool.post.0.\\1'),
                (r'^p_bn1.(.*)',            f'{12+num}_corner.layers.pool.post.1.\\1'),
                (r'^conv1.(.*)',            f'{12+num}_corner.layers.conv.0.\\1'),
                (r'^bn1.(.*)',              f'{12+num}_corner.layers.conv.1.\\1'),
                (r'^conv2.conv.(.*)',       f'{13+num}_convbatch.layers.0.\\1'),
                (r'^conv2.bn.(.*)',         f'{13+num}_convbatch.layers.1.\\1'),
            ]
        elif mod == 'heats':
            remap_detection = [
                (r'0.conv.(.*)',            f'output.heatmap.{14+num}_conv.\\1'),
                (r'1.(.*)',                 f'output.heatmap.{16+num}_conv.\\1'),
            ]
        elif mod == 'tags':
            remap_detection = [
                (r'0.conv.(.*)',            f'output.embedding.{17+num}_conv.\\1'),
                (r'1.(.*)',                 f'output.embedding.{19+num}_conv.\\1'),
            ]
        elif mod == 'offs':
            remap_detection = [
                (r'0.conv.(.*)',            f'output.offset.{20+num}_conv.\\1'),
                (r'1.(.*)',                 f'output.offset.{22+num}_conv.\\1'),
            ]

        for r in remap_detection:
            if re.match(r[0], rk) is not None:
                return nk + re.sub(r[0], r[1], rk)

        log.warn(f'Could not find matching layer for [{k}]')
        return None
