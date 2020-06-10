#
#   MobileNet classification network
#   Copyright EAVISE
#

from collections import OrderedDict
import functools
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['MobilenetV1']


class MobilenetV1(lnn.module.Lightnet):
    """ Mobilenet v1 classification network implementation :cite:`mobilenet_v1`.

    Args:
        num_classes (Number, optional): Number of classes; Default **1000**
        alpha (Number, optional): Number between [0-1] that controls the number of filters of the mobilenet convolutions; Default **1**
        input_channels (Number, optional): Number of input channels; Default **3**

    Attributes:
        self.inner_stride: Maximal internal subsampling factor of the network (input dimension should be a multiple of this)

    Note:
        The average pooling is implemented with an :class:`~torch.nn.AdaptiveAvgPool2d` layer. |br|
        For the base input dimension of 224x224, this is exactly the same as a 7x7 average pooling function,
        but the advantage of a adaptive average pooling is that this network can now handle multiple different input dimensions,
        as long as they are a multiple of the ``stride`` factor. |br|
        This is also how the implementation in `tensorflow <mobilenettf_>`_ (optionally) works.

    Warning:
        When changing the ``alpha`` value, you are changing the network architecture.
        This means you cannot use weights from this network with a different alpha value.

    .. _mobilenettf: https://github.com/tensorflow/models/blob/505f554c6417931c96b59516f14d1ad65df6dbc5/research/slim/nets/mobilenet_v1.py#L369-L378
    """
    inner_stride = 32

    def __init__(self, num_classes, alpha=1, input_channels=3):
        super().__init__()

        # Parameters
        self.num_classes = num_classes
        self.alpha = alpha
        self.input_channels = input_channels

        # Network
        relu = functools.partial(nn.ReLU6, inplace=True)
        self.layers = nn.Sequential(
            # Base layers
            nn.Sequential(OrderedDict([
                ('1_convbatch', lnn.layer.Conv2dBatchReLU(input_channels, int(alpha*32), 3, 2, 1, relu=relu)),
                ('2_convdw',    lnn.layer.Conv2dDepthWise(int(alpha*32), int(alpha*64), 3, 1, 1, relu=relu)),
                ('3_convdw',    lnn.layer.Conv2dDepthWise(int(alpha*64), int(alpha*128), 3, 2, 1, relu=relu)),
                ('4_convdw',    lnn.layer.Conv2dDepthWise(int(alpha*128), int(alpha*128), 3, 1, 1, relu=relu)),
                ('5_convdw',    lnn.layer.Conv2dDepthWise(int(alpha*128), int(alpha*256), 3, 2, 1, relu=relu)),
                ('6_convdw',    lnn.layer.Conv2dDepthWise(int(alpha*256), int(alpha*256), 3, 1, 1, relu=relu)),
                ('7_convdw',    lnn.layer.Conv2dDepthWise(int(alpha*256), int(alpha*512), 3, 2, 1, relu=relu)),
                ('8_convdw',    lnn.layer.Conv2dDepthWise(int(alpha*512), int(alpha*512), 3, 1, 1, relu=relu)),
                ('9_convdw',    lnn.layer.Conv2dDepthWise(int(alpha*512), int(alpha*512), 3, 1, 1, relu=relu)),
                ('10_convdw',   lnn.layer.Conv2dDepthWise(int(alpha*512), int(alpha*512), 3, 1, 1, relu=relu)),
                ('11_convdw',   lnn.layer.Conv2dDepthWise(int(alpha*512), int(alpha*512), 3, 1, 1, relu=relu)),
                ('12_convdw',   lnn.layer.Conv2dDepthWise(int(alpha*512), int(alpha*512), 3, 1, 1, relu=relu)),
                ('13_convdw',   lnn.layer.Conv2dDepthWise(int(alpha*512), int(alpha*1024), 3, 2, 1, relu=relu)),
                ('14_convdw',   lnn.layer.Conv2dDepthWise(int(alpha*1024), int(alpha*1024), 3, 1, 1, relu=relu)),
            ])),

            # Classification specific layers
            nn.Sequential(OrderedDict([
                ('15_avgpool',  nn.AdaptiveAvgPool2d(1)),
                ('16_dropout',  nn.Dropout()),
                ('17_conv',     nn.Conv2d(int(alpha*1024), num_classes, 1, 1, 0)),
                ('18_flatten',  lnn.layer.Flatten()),
            ])),
        )
