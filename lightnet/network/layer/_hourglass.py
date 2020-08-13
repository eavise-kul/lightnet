#
#   Cornernet HourGlass Module
#   Copyright EAVISE
#

from collections import OrderedDict
import torch.nn as nn
from ._util import Residual, ParallelSum

__all__ = ['HourGlass']


class HourGlass(nn.Module):
    """ Fully modifiable HourGlass module. |br|
    The default implementation follows that of :cite:`hourglass`, with singular residual blocks between each group,
    which we adapted from `their repository <hgrepo_>`_ to the best of our abilities.
    However, you can completely overhaul this module to fit your needs,
    by supplying your own "make" functions for each of the seperate blocks.

    Args:
        order (int): The number of recursive modules
        channels (int or list of ints): Channels for each of the recursive modules
        make_upper (function, optional): Function that should return a :class:`torch.nn.Module` for the upper part; Default **ResidualBlock**
        upper_kwargs (dict or list of dicts): Extra kwargs that are passed to the `make_upper` function; Default **None**
        make_down1 (function, optional): Function that should return a :class:`torch.nn.Module` for the first downward part; Default **2x2 MaxPool + ResidualBlock**
        down1_kwargs (dict or list of dicts): Extra kwargs that are passed to the `make_down1` function; Default **None**
        make_inner (function, optional): Function that should return a :class:`torch.nn.Module` for the inner part; Default **ResidualBlock**
        inner_kwargs (dict or list of dicts): Extra kwargs that are passed to the `make_inner` function; Default **None**
        make_down2 (function, optional): Function that should return a :class:`torch.nn.Module` for the second downward part; Default **ResidualBlock + 2x2 Upsampling (NN)**
        down2_kwargs (dict or list of dicts): Extra kwargs that are passed to the `make_down2` function; Default **None**

    .. figure:: /.static/api/hourglass.*
       :width: 100%
       :alt: HourGlass module design

       The HourGlass module consists of 4 different blocks: upper, down1, inner, down2. |br|
       You can modify each of these blocks by providing the correct make function and (optionally) keyword arguments.

    Note:
        All "make" functions have the following signature: ``function(in_channels, out_channels, **kwargs)``.

        The `in_channels` and `out_channels` get computed depending on the block (up, down1, down2 or inner), the current `order` and corresponding number of `channels`. |br|
        The keyword arguments of the different "make" functions are taken from the matching `..._kwargs` arguments,
        which can either be a fixed dictionary which will be used each time the function gets called,
        or a list of dictionaries, with different values for each recursion of the HourGlass module.
        If you pass a list of dictionaries, it should contain `order` number of dictionaries.

    Note:
        If you pass a list of integers as `channels`, you need to enter `order + 1` integers.
        The last recursive HourGlass module, wil use this extra channel number for its `inner` module,
        allowing to use the same `down1` and `down2` modules in each HourGlass. |br|
        If you pass a single integer, it will be used for all modules.

    Example:
        >>> module = ln.network.layer.HourGlass(
        ...     2,                  # 2 nested hourglasses
        ...     [3, 32, 64],        # 2+1 channels, one for each hourglass level and a last one for the inner module
        ...
        ...     # UPPER: single convbatchrelu
        ...     make_upper=lambda ci, co: ln.network.layer.Conv2dBatchReLU(ci, co, 3, 1, 1),
        ...
        ...     # DOWN 1: Providing keyword arguments
        ...     make_down1=lambda ci, co, **kwargs: ln.network.layer.Conv2dBatchReLU(ci, co, **kwargs),
        ...     down1_kwargs={'kernel_size': 3, 'stride': 2, 'padding': 1},
        ...
        ...     # INNER: Conv2dDepthWise
        ...     make_inner=lambda ci, co: ln.network.layer.Conv2dDepthWise(ci, co, 3, 1, 1),
        ...
        ...     # DOWN 2: Separate kwargs for each hourglass level
        ...     make_down2=lambda ci, co, **kwargs: torch.nn.Sequential(
        ...         ln.network.layer.Conv2dBatchReLU(ci, co, **kwargs),
        ...         torch.nn.Upsample(scale_factor=2, mode='nearest'),
        ...     ),
        ...     down2_kwargs=[{'kernel_size': 3, 'stride': 1, 'padding': 1}, {'kernel_size': 1, 'stride': 1, 'padding': 0}],
        ... )
        >>> print(module)
        HourGlass(
          (layers): ParallelSum(
            (upper): Conv2dBatchReLU(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (down): Sequential(
              (down1): Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), ReLU(inplace=True))
              (inner): HourGlass(
                (layers): ParallelSum(
                  (upper): Conv2dBatchReLU(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
                  (down): Sequential(
                    (down1): Conv2dBatchReLU(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), ReLU(inplace=True))
                    (inner): Conv2dDepthWise(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
                    (down2): Sequential(
                      (0): Conv2dBatchReLU(64, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), ReLU(inplace=True))
                      (1): Upsample(scale_factor=2.0, mode=nearest)
                    )
                  )
                )
              )
              (down2): Sequential(
                (0): Conv2dBatchReLU(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
                (1): Upsample(scale_factor=2.0, mode=nearest)
              )
            )
          )
        )

    .. _hgrepo: https://github.com/princeton-vl/pose-hg-train
    """
    def __init__(
        self,
        order,
        channels,
        make_upper=None,
        upper_kwargs=None,
        make_down1=None,
        down1_kwargs=None,
        make_inner=None,
        inner_kwargs=None,
        make_down2=None,
        down2_kwargs=None,
    ):
        super().__init__()

        make_upper = make_upper if make_upper is not None else self._make_upper
        make_down1 = make_down1 if make_down1 is not None else self._make_down1
        make_inner = make_inner if make_inner is not None else self._make_inner
        make_down2 = make_down2 if make_down2 is not None else self._make_down2
        c1 = self._get_current(channels)
        c2 = self._get_current(self._get_next(channels))

        if order > 1:
            inner = HourGlass(
                order - 1,
                self._get_next(channels),
                make_upper,
                self._get_next(upper_kwargs),
                make_down1,
                self._get_next(down1_kwargs),
                make_inner,
                self._get_next(inner_kwargs),
                make_down2,
                self._get_next(down2_kwargs),
            )
        else:
            inner = make_inner(c2, c2, **self._get_current(inner_kwargs, dict()))

        self.layers = ParallelSum(OrderedDict([
            ('upper', make_upper(c1, c1, **self._get_current(upper_kwargs, dict()))),
            ('down', nn.Sequential(OrderedDict([
                ('down1', make_down1(c1, c2, **self._get_current(down1_kwargs, dict()))),
                ('inner', inner),
                ('down2', make_down2(c2, c1, **self._get_current(down2_kwargs, dict()))),
            ]))),
        ]))

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def _make_upper(in_channels, out_channels, **kwargs):
        return get_residual(in_channels, out_channels)

    @staticmethod
    def _make_down1(in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.MaxPool2d(2, 2),
            get_residual(in_channels, out_channels),
        )

    @staticmethod
    def _make_inner(in_channels, out_channels, **kwargs):
        return get_residual(in_channels, out_channels)

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
