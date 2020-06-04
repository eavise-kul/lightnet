#
#   Cornernet HourGlass Module
#   Copyright EAVISE
#

from collections import OrderedDict
import torch.nn as nn
from ._util import Residual, SumSequential

__all__ = ['HourGlass']


class HourGlass(nn.Module):
    """ Fully modifiable HourGlass module. |br|
    The default implementation follows that of :cite:`hourglass`, with singular residual blocks between each group,
    but you can completely overhaul this module to fit your needs, by supplying your own "make" functions.

    .. figure:: ../.static/hourglass.png
       :width: 100%
       :alt: HourGlass module design

       The HourGlass module consists of 4 different blocks: up, down1, inner, down2. |br|
       You can modify each of these blocks by providing the correct make function and (optionally) keyword arguments.

    Args:
        order (int): The number of recursive modules
        channels (int or list of ints): Channels for each of the recursive modules (see Note)
        make_up (function, optional): Function that should return a :class:`torch.nn.Module` for the upper part; Default **Single Residual**
        up_kwargs (dict or list of dicts) extra kwargs that are passed to the `make_up` function (see Note); Default **None**
        make_down1 (function, optional): Function that should return a :class:`torch.nn.Module` for the first downward part; Default **2x2 MaxPool + Residual**
        down1_kwargs (dict or list of dicts) extra kwargs that are passed to the `make_down1` function (see Note); Default **None**
        make_inner (function, optional): Function that should return a :class:`torch.nn.Module` for the inner part; Default **Single Residual**
        inner_kwargs (dict or list of dicts) extra kwargs that are passed to the `make_inner` function (see Note); Default **None**
        make_down2 (function, optional): Function that should return a :class:`torch.nn.Module` for the second downward part; Default **Single Residual + 2x2 Upsampling (NN)**
        down2_kwargs (dict or list of dicts) extra kwargs that are passed to the `make_down2` function (see Note); Default **None**

    Note:
        If you pass a list of integers as `channels`, you need to enter `order + 1` integers.
        The last recursive HourGlass module, wil use this extra channel number for its `inner` module,
        allowing to use the same `down1` and `down2` modules in each HourGlass. |br|
        If you pass a single integer, it will be used for all modules.

    Note:
        The keyword arguments of the different "make" functions can either be a fixed dictionary which will be used each time the function gets called,
        or it can be a list of dictionaries, with different values for each recursion of the HourGlass module. |br|
        If you pass a list of dictionaries, it should contain `order` dictionaries.

    Note:
        The default "make" functions create residual blocks as explained in :cite:`hourglass`. |br|
        We adapted the code from `their repository <hgrepo_>`_ to the best of our abilities.

    .. _hgrepo: https://github.com/princeton-vl/pose-hg-train
    """
    def __init__(
        self,
        order,
        channels,
        make_up=None,
        up_kwargs=None,
        make_down1=None,
        down1_kwargs=None,
        make_inner=None,
        inner_kwargs=None,
        make_down2=None,
        down2_kwargs=None,
    ):
        super().__init__()

        make_up = make_up if make_up is not None else self._make_up
        make_down1 = make_down1 if make_down1 is not None else self._make_down1
        make_inner = make_inner if make_inner is not None else self._make_inner
        make_down2 = make_down2 if make_down2 is not None else self._make_down2
        c1 = self._get_current(channels)
        c2 = self._get_current(self._get_next(channels))

        if order > 1:
            inner = HourGlass(
                order - 1,
                self._get_next(channels),
                make_up,
                self._get_next(up_kwargs),
                make_down1,
                self._get_next(down1_kwargs),
                make_inner,
                self._get_next(inner_kwargs),
                make_down2,
                self._get_next(down2_kwargs),
            )
        else:
            inner = make_inner(c2, **self._get_current(inner_kwargs, dict()))

        self.layers = SumSequential(OrderedDict([
            ('up', make_up(c1, **self._get_current(up_kwargs, dict()))),
            ('down', nn.Sequential(OrderedDict([
                ('down1', make_down1(c1, c2, **self._get_current(down1_kwargs, dict()))),
                ('inner', inner),
                ('down2', make_down2(c2, c1, **self._get_current(down2_kwargs, dict()))),
            ]))),
        ]))

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def _make_up(channels, **kwargs):
        return get_residual(channels, channels)

    @staticmethod
    def _make_down1(in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.MaxPool2d(2, 2),
            get_residual(in_channels, out_channels),
        )

    @staticmethod
    def _make_inner(channels, **kwargs):
        return get_residual(channels, channels)

    @staticmethod
    def _make_down2(in_channels, out_channels, **kwargs):
        return nn.Sequential(
            get_residual(in_channels, out_channels),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

    @staticmethod
    def _get_current(value, default=None):
        if value is None:
            return default
        elif isinstance(value, (list, tuple)):
            return value[0] if len(value) >= 1 else default
        else:
            return value

    @staticmethod
    def _get_next(value):
        if isinstance(value, (list, tuple)):
            return value[1:]
        else:
            return value


def get_residual(in_channels, out_channels):
    """ Basic residual block implementation from :cite:`hourglass`. """
    return Residual(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels//2, 1, 1, 0),
        nn.BatchNorm2d(out_channels//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels//2, out_channels//2, 3, 1, 1),
        nn.BatchNorm2d(out_channels//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels//2, out_channels, 1, 1, 0),

        skip=None if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, 1, 0)
    )
