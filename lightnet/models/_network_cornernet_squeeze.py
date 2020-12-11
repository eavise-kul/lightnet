#
#   Cornernet Squeeze model
#   Copyright EAVISE
#

import functools
import logging
import re
from collections import OrderedDict
import torch
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['CornernetSqueeze']
log = logging.getLogger('lightnet.models')


class CornernetSqueeze(lnn.module.Lightnet):
    """ Cornernet Squeeze implementation :cite:`cornernet_lite`.

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
    stride = 8
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
            ('3_residual',                  self.get_residual(256, 256, stride=2)),
            ('para',                        lnn.layer.ParallelSum(
                lnn.layer.SequentialSelect(['5_convbatch'], False, OrderedDict([
                    ('4_hourglass',         self.get_hourglass(self.get_fire)),
                    ('5_convbatch',         lnn.layer.Conv2dBatchReLU(256, 256, 3, 1, 1)),
                    ('6_conv',              nn.Conv2d(256, 256, 1, 1, 0, bias=False)),
                    ('7_batchnorm',         nn.BatchNorm2d(256)),
                ])),
                nn.Sequential(OrderedDict([
                    ('8_conv',              nn.Conv2d(256, 256, 1, 1, 0, bias=False)),
                    ('9_batchnorm',         nn.BatchNorm2d(256)),
                ])),
                post=nn.ReLU(inplace=True),
            )),
            ('10_residual',                  self.get_residual(256, 256)),
            ('11_hourglass',                self.get_hourglass(self.get_fire)),
            ('12_convbatch',                lnn.layer.Conv2dBatchReLU(256, 256, 3, 1, 1))
        ]))

        self.detector = lnn.layer.ParallelCat(OrderedDict([
            ('topleft',                     nn.Sequential(OrderedDict([
                ('13_corner',               lnn.layer.CornerPool(256, lnn.layer.TopPool, lnn.layer.LeftPool)),
                ('14_convbatch',            lnn.layer.Conv2dBatchReLU(256, 256, 3, 1, 1)),
                ('output',                  lnn.layer.ParallelCat(OrderedDict([
                    ('heatmap',             nn.Sequential(OrderedDict([
                        ('15_conv',         nn.Conv2d(256, 256, 1, 1, 0, bias=True)),
                        ('16_relu',         nn.ReLU(inplace=True)),
                        ('17_conv',         nn.Conv2d(256, num_classes, 1, 1, 0)),
                    ]))),
                    ('embedding',           nn.Sequential(OrderedDict([
                        ('18_conv',         nn.Conv2d(256, 256, 1, 1, 0, bias=True)),
                        ('19_relu',         nn.ReLU(inplace=True)),
                        ('20_conv',         nn.Conv2d(256, 1, 1, 1, 0)),
                    ]))),
                    ('offset',              nn.Sequential(OrderedDict([
                        ('21_conv',         nn.Conv2d(256, 256, 1, 1, 0, bias=True)),
                        ('22_relu',         nn.ReLU(inplace=True)),
                        ('23_conv',         nn.Conv2d(256, 2, 1, 1, 0)),
                    ]))),
                ]))),
            ]))),
            ('bottomright',                 nn.Sequential(OrderedDict([
                ('24_corner',               lnn.layer.CornerPool(256, lnn.layer.BottomPool, lnn.layer.RightPool)),
                ('25_convbatch',            lnn.layer.Conv2dBatchReLU(256, 256, 3, 1, 1)),
                ('output',                  lnn.layer.ParallelCat(OrderedDict([
                    ('heatmap',             nn.Sequential(OrderedDict([
                        ('26_conv',         nn.Conv2d(256, 256, 1, 1, 0, bias=True)),
                        ('27_relu',         nn.ReLU(inplace=True)),
                        ('28_conv',         nn.Conv2d(256, num_classes, 1, 1, 0)),
                    ]))),
                    ('embedding',           nn.Sequential(OrderedDict([
                        ('29_conv',         nn.Conv2d(256, 256, 1, 1, 0, bias=True)),
                        ('30_relu',         nn.ReLU(inplace=True)),
                        ('31_conv',         nn.Conv2d(256, 1, 1, 1, 0)),
                    ]))),
                    ('offset',              nn.Sequential(OrderedDict([
                        ('32_conv',         nn.Conv2d(256, 256, 1, 1, 0, bias=True)),
                        ('33_relu',         nn.ReLU(inplace=True)),
                        ('34_conv',         nn.Conv2d(256, 2, 1, 1, 0)),
                    ]))),
                ]))),
            ]))),
        ]))

        if not self.inference_only:
            self.intermediate = lnn.layer.ParallelCat(OrderedDict([
                ('topleft',                     nn.Sequential(OrderedDict([
                    ('35_corner',               lnn.layer.CornerPool(256, lnn.layer.TopPool, lnn.layer.LeftPool)),
                    ('36_convbatch',            lnn.layer.Conv2dBatchReLU(256, 256, 3, 1, 1)),
                    ('output',                  lnn.layer.ParallelCat(OrderedDict([
                        ('heatmap',             nn.Sequential(OrderedDict([
                            ('37_conv',         nn.Conv2d(256, 256, 1, 1, 0, bias=True)),
                            ('38_relu',         nn.ReLU(inplace=True)),
                            ('39_conv',         nn.Conv2d(256, num_classes, 1, 1, 0)),
                        ]))),
                        ('embedding',           nn.Sequential(OrderedDict([
                            ('40_conv',         nn.Conv2d(256, 256, 1, 1, 0, bias=True)),
                            ('41_relu',         nn.ReLU(inplace=True)),
                            ('42_conv',         nn.Conv2d(256, 1, 1, 1, 0)),
                        ]))),
                        ('offset',              nn.Sequential(OrderedDict([
                            ('43_conv',         nn.Conv2d(256, 256, 1, 1, 0, bias=True)),
                            ('44_relu',         nn.ReLU(inplace=True)),
                            ('45_conv',         nn.Conv2d(256, 2, 1, 1, 0)),
                        ]))),
                    ]))),
                ]))),
                ('bottomright',                 nn.Sequential(OrderedDict([
                    ('46_corner',               lnn.layer.CornerPool(256, lnn.layer.BottomPool, lnn.layer.RightPool)),
                    ('47_convbatch',            lnn.layer.Conv2dBatchReLU(256, 256, 3, 1, 1)),
                    ('output',                  lnn.layer.ParallelCat(OrderedDict([
                        ('heatmap',             nn.Sequential(OrderedDict([
                            ('48_conv',         nn.Conv2d(256, 256, 1, 1, 0, bias=True)),
                            ('49_relu',         nn.ReLU(inplace=True)),
                            ('50_conv',         nn.Conv2d(256, num_classes, 1, 1, 0)),
                        ]))),
                        ('embedding',           nn.Sequential(OrderedDict([
                            ('51_conv',         nn.Conv2d(256, 256, 1, 1, 0, bias=True)),
                            ('52_relu',         nn.ReLU(inplace=True)),
                            ('53_conv',         nn.Conv2d(256, 1, 1, 1, 0)),
                        ]))),
                        ('offset',              nn.Sequential(OrderedDict([
                            ('54_conv',         nn.Conv2d(256, 256, 1, 1, 0, bias=True)),
                            ('55_relu',         nn.ReLU(inplace=True)),
                            ('56_conv',         nn.Conv2d(256, 2, 1, 1, 0)),
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
        out2 = self.intermediate(self.extractor[3][0].selected)
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
    def get_fire(in_channels, out_channels, kernel=3, stride=1, padding=1):
        layers = [
            nn.Conv2d(in_channels, out_channels // 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            lnn.layer.ParallelCat(
                nn.Conv2d(out_channels // 2, out_channels // 2, 1, stride, 0, bias=False),
                nn.Conv2d(out_channels // 2, out_channels // 2, kernel, stride, padding, bias=False, groups=out_channels // 2)
            ),
            nn.BatchNorm2d(out_channels)
        ]

        if stride == 1 and in_channels == out_channels:
            return lnn.layer.Residual(*layers, post=nn.ReLU(inplace=True))
        else:
            return nn.Sequential(*layers, nn.ReLU(inplace=True))

    @staticmethod
    def get_hourglass(residual):
        return lnn.layer.HourGlass(
            4, [256, 256, 384, 384, 512],
            make_upper=lambda ci, co: nn.Sequential(*[residual(ci, co) for _ in range(2)]),
            make_down1=lambda ci, co: nn.Sequential(residual(ci, co, stride=2), residual(co, co)),
            make_inner=lambda ci, co: nn.Sequential(*[residual(ci, co) for _ in range(4)]),
            make_down2=lambda ci, co: nn.Sequential(residual(ci, ci), residual(ci, co), nn.ConvTranspose2d(co, co, 4, 2, 1))
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
            (r'^module.hg.pre.2.conv1.(.*)',    r'extractor.3_residual.0.\1'),
            (r'^module.hg.pre.2.bn1.(.*)',      r'extractor.3_residual.1.\1'),
            (r'^module.hg.pre.2.conv2.(.*)',    r'extractor.3_residual.3.\1'),
            (r'^module.hg.pre.2.bn2.(.*)',      r'extractor.3_residual.4.\1'),
            (r'^module.hg.pre.2.skip.(.*)',     r'extractor.3_residual.skip.\1'),
            (r'^module.hg.cnvs.0.conv.(.*)',    r'extractor.para.0.5_convbatch.layers.0.\1'),
            (r'^module.hg.cnvs.0.bn.(.*)',      r'extractor.para.0.5_convbatch.layers.1.\1'),
            (r'^module.hg.cnvs.1.conv.(.*)',    r'extractor.12_convbatch.layers.0.\1'),
            (r'^module.hg.cnvs.1.bn.(.*)',      r'extractor.12_convbatch.layers.1.\1'),
            (r'^module.hg.cnvs_.0.0.(.*)',      r'extractor.para.0.6_conv.\1'),
            (r'^module.hg.cnvs_.0.1.(.*)',      r'extractor.para.0.7_batchnorm.\1'),
            (r'^module.hg.inters.0.conv1.(.*)', r'extractor.10_residual.0.\1'),
            (r'^module.hg.inters.0.bn1.(.*)',   r'extractor.10_residual.1.\1'),
            (r'^module.hg.inters.0.conv2.(.*)', r'extractor.10_residual.3.\1'),
            (r'^module.hg.inters.0.bn2.(.*)',   r'extractor.10_residual.4.\1'),
            (r'^module.hg.inters.0.skip.(.*)',  r'extractor.10_residual.skip.\1'),
            (r'^module.hg.inters_.0.0.(.*)',    r'extractor.para.1.8_conv.\1'),
            (r'^module.hg.inters_.0.1.(.*)',    r'extractor.para.1.9_batchnorm.\1'),
        ]
        for r in remap_extractor:
            if re.match(r[0], k) is not None:
                return re.sub(r[0], r[1], k)

        # HOURGLASSES
        if k.startswith('module.hg.hgs'):
            if k[14] == '0':
                nk = 'extractor.para.0.4_hourglass.layers.'
            else:
                nk = 'extractor.11_hourglass.layers.'

            k = (
                k[16:]
                .replace('up1', 'upper')
                .replace('up2', 'down.down2.2')
                .replace('low1', 'down.down1')
                .replace('low3', 'down.down2')
                .replace('conv1', '0')
                .replace('bn1', '1')
                .replace('conv_1x1', '2.0')
                .replace('conv_3x3', '2.1')
                .replace('bn2', '3')
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
                (r'^p1_conv1.conv.(.*)',    f'{13+num}_corner.layers.pool.0.0.layers.0.\\1'),
                (r'^p1_conv1.bn.(.*)',      f'{13+num}_corner.layers.pool.0.0.layers.1.\\1'),
                (r'^p2_conv1.conv.(.*)',    f'{13+num}_corner.layers.pool.1.0.layers.0.\\1'),
                (r'^p2_conv1.bn.(.*)',      f'{13+num}_corner.layers.pool.1.0.layers.1.\\1'),
                (r'^p_conv1.(.*)',          f'{13+num}_corner.layers.pool.post.0.\\1'),
                (r'^p_bn1.(.*)',            f'{13+num}_corner.layers.pool.post.1.\\1'),
                (r'^conv1.(.*)',            f'{13+num}_corner.layers.conv.0.\\1'),
                (r'^bn1.(.*)',              f'{13+num}_corner.layers.conv.1.\\1'),
                (r'^conv2.conv.(.*)',       f'{14+num}_convbatch.layers.0.\\1'),
                (r'^conv2.bn.(.*)',         f'{14+num}_convbatch.layers.1.\\1'),
            ]
        elif mod == 'heats':
            remap_detection = [
                (r'0.conv.(.*)',            f'output.heatmap.{15+num}_conv.\\1'),
                (r'1.(.*)',                 f'output.heatmap.{17+num}_conv.\\1'),
            ]
        elif mod == 'tags':
            remap_detection = [
                (r'0.conv.(.*)',            f'output.embedding.{18+num}_conv.\\1'),
                (r'1.(.*)',                 f'output.embedding.{20+num}_conv.\\1'),
            ]
        elif mod == 'offs':
            remap_detection = [
                (r'0.conv.(.*)',            f'output.offset.{21+num}_conv.\\1'),
                (r'1.(.*)',                 f'output.offset.{23+num}_conv.\\1'),
            ]

        for r in remap_detection:
            if re.match(r[0], rk) is not None:
                return nk + re.sub(r[0], r[1], rk)

        log.warn(f'Could not find matching layer for [{k}]')
        return None
