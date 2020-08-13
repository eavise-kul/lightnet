#
#   Fusion module
#   Copyright EAVISE
#

import copy
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent


__all__ = ['Fusion']
log = logging.getLogger(__name__)


class Fusion(nn.Module):
    """ Fusion module from :cite:`rgbd_fusion_v2`.
    This module is like a :class:`torch.nn.Sequential`, but it will perform the actions twice,
    once on the regular input and once on the fusion input. |br|
    The fusion will be performed by adding an extra 1x1 fuse convolution between the output of both streams and the input of the combined stream,
    to mix both streams and reduce the number of output feature maps by a factor of 2. |br|
    This module takes a single input feature map during its forward pass, and splits it evenly for both input streams.

    Args:
        layers (dict or list of pytorch modules): Layers that will be used. These layers are internally passed to a :class:`~torch.nn.Sequential` and must thus comply with the rules for this class
        fuse_layer (int, optional): Number between 0 and the number of layers + 1, that controls where to fuse both streams; Default **None**

    .. figure:: /.static/api/fusion.*
       :width: 100%
       :alt: Fusion module and fuse layer

    Note:
        Depending on the value of the `fuse_layer` attribute, fusion is performed at different stages of the module. |br|

        - If no `fuse\\_layer` is given (or **None** is given as its value),\
        no fusion will be done and the input will be considered as an already fused combination.
        - If the `fuse\\_layer` attribute is an integer from 0 to :math:`num\\_layers`, the module will fuse both streams after the number of the layer that is given.\
        Giving a value of **0** thus means to fuse before the first layer and giving a value of **num_layers** to fuse after the last.
        - Finally, if :math:`fuse\\_layer == num\\_layers + 1`, then no fusion will occur, but rather both streams will be processed seperately\
        and the output feature maps will simply be concatenated at the end.

        These rules allow the chain multiple :class:`~lightnet.network.layer.Fusion` modules together, only fusing in one of them at a certain time.

    Example:
        >>> layers = [
        ...   ln.network.layer.Conv2dBatchReLU(3,  32, 3, 1, 1),
        ...   ln.network.layer.Conv2dBatchReLU(32, 64, 3, 1, 1),
        ...   ln.network.layer.Conv2dBatchReLU(64, 32, 3, 1, 1),
        ... ]
        >>> # Streams are fused in the middle of the module
        >>> module = ln.network.layer.Fusion(layers, fuse_layer=1)
        >>> print(module)
        Fusion(
          (Regular & Fusion): Sequential(
            (0): Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          )
          (Fuse): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (Combined): Sequential(
            (0): Conv2dBatchReLU(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (1): Conv2dBatchReLU(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          )
        )
        >>> # Streams are fused at the end of the module (after last convolution)
        >>> module = ln.network.layer.Fusion(layers, fuse_layer=3)
        >>> print(module)
        Fusion(
          (Regular & Fusion): Sequential(
            (0): Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (1): Conv2dBatchReLU(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (2): Conv2dBatchReLU(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          )
          (Fuse): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        >>> # Streams are fused before the first layer
        >>> module = ln.network.layer.Fusion(layers, fuse_layer=0)
        >>> print(module)
        Fusion(
          (Fuse): Conv2d(6, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (Combined): Sequential(
            (0): Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (1): Conv2dBatchReLU(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (2): Conv2dBatchReLU(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          )
        )
        >>> # Streams were fused before this module and thus there is no fusion involved (only combined sequential)
        >>> module = ln.network.layer.Fusion(layers, fuse_layer=None)
        >>> print(module)
        Fusion(
          (Combined): Sequential(
            (0): Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (1): Conv2dBatchReLU(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (2): Conv2dBatchReLU(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          )
        )
        >>> # Streams are not fused in this module (duplicated regular and fusion sequentials)
        >>> module = ln.network.layer.Fusion(layers, fuse_layer=4)
        >>> print(module)
        Fusion(
          (Regular & Fusion): Sequential(
            (0): Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (1): Conv2dBatchReLU(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (2): Conv2dBatchReLU(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          )
        )

    Warning:
        This module will create a :class:`~torch.nn.Sequential` for the regular convolutional stream and deepcopy that for the fusion stream.
        This will effectively create 2 different streams that have their own weights, but it does mean that both streams start with identical weights. |br|
        It is strongly advised to use pretrained weights or initialize your weights randomly after having created these modules.

    Warning:
        The way we compute the input and output feature maps for the 1x1 fuse convolution,
        is by looping through the regular stream or combined stream,
        looking for the last `out_channels` or first `in_channels` attribute of the layers respectively. |br|
        This means that this module only works if there are convolutional layers in the list,
        or any other layer that has these `in_channels` and `out_channels` attributes to be able to deduct the number of feature maps.
    """
    def __init__(self, layers, fuse_layer=None):
        super().__init__()

        # Parameters
        self.fuse_layer = fuse_layer

        # layers
        if self.fuse_layer is None:             # Combined
            self.regular = None
            self.fusion = None
            if isinstance(layers, dict):
                self.combined = nn.Sequential(layers)
            else:
                self.combined = nn.Sequential(*layers)
            self.fuse = None
        elif self.fuse_layer == 0:              # Fuse + Combined
            self.regular = None
            self.fusion = None
            if isinstance(layers, dict):
                self.combined = nn.Sequential(layers)
            else:
                self.combined = nn.Sequential(*layers)
            self.fuse = self._get_fuse_conv()
        elif self.fuse_layer == len(layers):    # Reg/Fusion + Fuse
            if isinstance(layers, dict):
                self.regular = nn.Sequential(layers)
            else:
                self.regular = nn.Sequential(*layers)
            self.fusion = copy.deepcopy(self.regular)
            self.combined = None
            self.fuse = self._get_fuse_conv()
        elif self.fuse_layer > len(layers):     # Reg/Fusion
            if self.fuse_layer > len(layers) + 1:
                log.debug(f'fuse_layer variable is too high, setting it to {len(layers)+1} which will not perform any fusion [{self.fuse_layer}/{len(layers)+1}]')
                self.fuse_layer = len(layers) + 1

            if isinstance(layers, dict):
                self.regular = nn.Sequential(layers)
            else:
                self.regular = nn.Sequential(*layers)
            self.fusion = copy.deepcopy(self.regular)
            self.combined = None
            self.fuse = None
        elif self.fuse_layer < len(layers):     # Reg/Fusion + Fuse + Combined
            if isinstance(layers, dict):
                self.regular = nn.Sequential(OrderedDict(list(layers.items())[:self.fuse_layer]))
                self.fusion = copy.deepcopy(self.regular)
                self.combined = nn.Sequential(OrderedDict(list(layers.items())[self.fuse_layer:]))
            else:
                self.regular = nn.Sequential(*layers[:self.fuse_layer])
                self.fusion = copy.deepcopy(self.regular)
                self.combined = nn.Sequential(*layers[self.fuse_layer:])
            self.fuse = self._get_fuse_conv()

    def __repr__(self):
        main_str = self._get_name() + '('

        if self.regular is not None:
            mod_str = _addindent(repr(self.regular), 2)
            main_str += '\n  (Regular & Fusion): ' + mod_str
        if self.fuse is not None:
            mod_str = _addindent(repr(self.fuse), 2)
            main_str += '\n  (Fuse): ' + mod_str
        if self.combined is not None:
            mod_str = _addindent(repr(self.combined), 2)
            main_str += '\n  (Combined): ' + mod_str

        return main_str + '\n)'

    def _get_fuse_conv(self):
        channels = None
        if self.combined is not None:
            channels = find_attr(self.combined, 'in_channels')
        if channels is None and self.regular is not None:
            channels = find_attr(self.regular, 'out_channels', first=False)
        if channels is None:
            raise TypeError('Could not find "in_channels" or "out_channels" attribute in layers.')

        return nn.Conv2d(channels*2, channels, 1, 1, 0, bias=False).to(list(self.parameters())[0].device)

    def forward(self, x):
        if self.regular is not None:
            channels = x.size(1)
            if channels % 2 != 0:
                raise ValueError(f'Number of input channels is not divisible by 2 [{channels}]')

            r = self.regular(x[:, :channels//2])
            f = self.fusion(x[:, channels//2:])
            x = torch.cat((r, f), 1)

        if self.fuse is not None:
            x = self.fuse(x)

        if self.combined is not None:
            x = self.combined(x)

        return x


def find_attr(module, name, first=True):
    if hasattr(module, name):
        return getattr(module, name)

    retval = None
    for mod in module.children():
        r = find_attr(mod, name, first)
        if r is not None:
            if first:
                return r
            else:
                retval = r

    return retval
