#
#   Utility layers
#   Copyright EAVISE
#

import logging
import torch
import torch.nn as nn


__all__ = ['Residual', 'SelectiveSequential', 'SumSequential']
log = logging.getLogger(__name__)


class Residual(nn.Sequential):
    """ Residual block that runs like a Sequential, but then adds the original input to the output tensor.
        See :class:`torch.nn.Sequential` for more information.

        Warning:
            The dimension between the input and output of the module need to be the same
            or need to be broadcastable from one to the other!
    """
    def forward(self, x):
        y = super().forward(x)
        return x + y


class SelectiveSequential(nn.Sequential):
    """ Sequential that allows to select which layers are to be considered as output.
        See :class:`torch.nn.Sequential` for more information.

        Args:
            selection (list): names of the layers for which you want to get the output
            *args: Arguments that are passed to the Sequential init function

        Note:
            If your selection only contains one item, this layer will flatten the return output in a single list:

            >>> main_output, selected_output = layer(input)  # doctest: +SKIP

            However, if you have multiple outputs in your selection list, the outputs will be grouped in a dictionary:

            >>> main_output, selected_output_dict = layer(input)  # doctest: +SKIP
    """
    def __init__(self, selection, *args):
        super().__init__(*args)

        self.selection = [str(select) for select in selection]
        self.flatten = len(self.selection) == 1
        k = list(self._modules.keys())
        for sel in self.selection:
            if sel not in k:
                raise KeyError('Selection key not found in sequential [{sel}]')

    def extra_repr(self):
        return f'selection={self.selection}, flatten={self.flatten}'

    def forward(self, x):
        sel_output = {sel: None for sel in self.selection}

        for key, module in self._modules.items():
            x = module(x)
            if key in self.selection:
                sel_output[key] = x

        if self.flatten:
            return x, sel_output[self.selection[0]]
        else:
            return x, sel_output


class SumSequential(nn.Sequential):
    """ Sequential container that runs each module on the input,
    and combines the different outputs by summing them.

    Args:
        *args: Arguments passed to :class:`torch.nn.Sequential`
        post (nn.Module, optional): Extra module that is run on the sum of the outputs of the other modules; Default **None**

    Note:
        If you are using an `OrderedDict` to pass the modules to the sequential,
        you can set the `post` value inside of that dict as well.
    """
    def __init__(self, *args, post=None):
        super().__init__(*args)
        if post is None and 'post' in dir(self):
            self.post = self._modules['post']
        else:
            self.post = post

    def forward(self, x):
        output = torch.sum(
            torch.stack([module(x) for name, module in self.named_children() if name != 'post']),
            dim=0
        )

        if self.post is not None:
            output = self.post(output)

        return output
