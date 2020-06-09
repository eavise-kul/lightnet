#
#   Utility layers
#   Copyright EAVISE
#

from collections import OrderedDict
import logging
import torch
import torch.nn as nn


__all__ = ['Residual', 'SequentialSelect', 'Parallel', 'ParallelCat', 'ParallelSum']
log = logging.getLogger(__name__)


class Residual(nn.Sequential):
    """ Residual block that runs like a Sequential, but then adds the original input to the output tensor.
    See :class:`torch.nn.Sequential` for more information.

    Args:
        *args: Arguments passed to :class:`torch.nn.Sequential`
        skip (nn.Module, optional): Extra module that is run on the input before adding it to the main block; Default **None**
        post (nn.Module, optional): Extra module that is run on the output after everything is added; Default **None**

    Note:
        If you are using an OrderedDict to pass the modules to the sequential,
        you can set the `skip` and `post` values inside of that dict as well.

    Warning:
        The dimension between the input (after skip) and output of the module need to be the same
        or need to be broadcastable from one to the other!
    """
    def __init__(self, *args, skip=None, post=None):
        super().__init__(*args)
        if skip is None and 'skip' in dir(self):
            self.skip = self._modules['skip']
        else:
            self.skip = skip

        if post is None and 'post' in dir(self):
            self.post = self._modules['post']
        else:
            self.post = post

    def forward(self, x):
        y = x
        for name, module in self.named_children():
            if name not in ('skip', 'post'):
                y = module(y)

        if self.skip is not None:
            x = self.skip(x)

        z = x + y
        if self.post is not None:
            z = self.post(z)

        return z


class SequentialSelect(nn.Sequential):
    """ Sequential that allows to select which layers are to be considered as output.
        See :class:`torch.nn.Sequential` for more information.

        Args:
            selection (list): names of the layers for which you want to get the output
            *args: Arguments that are passed to the Sequential init function
            return_selection (bool): Whether to return the selected layers, or just store them as `self.selected`

        Note:
            If your selection only contains one item, this layer will flatten the return output in a single list:

            >>> main_output, selected_output = layer(input)  # doctest: +SKIP

            However, if you have multiple outputs in your selection list, the outputs will be grouped in a dictionary:

            >>> main_output, selected_output_dict = layer(input)  # doctest: +SKIP

            The same applies for `self.selected` in case `return_selection` is set to False.

        Note:
            The `return_selection` argument can optionally also be passed as the first or last of *args.
    """
    def __init__(self, selection, *args, return_selection=True):
        if isinstance(args[0], bool):
            return_selection = args[0]
            args = args[1:]
        elif isinstance(args[-1], bool):
            return_selection = args[-1]
            args = args[:-1]

        super().__init__(*args)

        self.return_selection = return_selection
        self.selected = None
        self.selection = [str(select) for select in selection]
        self.flatten = len(self.selection) == 1
        k = list(self._modules.keys())
        for sel in self.selection:
            if sel not in k:
                raise KeyError('Selection key not found in sequential [{sel}]')

    def extra_repr(self):
        return f'selection={self.selection}, return={self.return_selection}, flatten={self.flatten}'

    def forward(self, x):
        sel_output = {sel: None for sel in self.selection}

        for key, module in self._modules.items():
            x = module(x)
            if key in self.selection:
                sel_output[key] = x

        # Return
        if not self.return_selection:
            if self.flatten:
                self.selected = sel_output[self.selection[0]]
            else:
                self.selected = sel_output

            return x

        if self.flatten:
            return x, sel_output[self.selection[0]]
        else:
            return x, sel_output


class Parallel(nn.Sequential):
    """ Container that runs each module on the input.
    The ouput is a list that contains the output of each of the different modules.

    Args:
        *args: Modules to run in parallel (similar to :class:`torch.nn.Sequential`)
    """
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        return [module(x) for module in self]


class ParallelCat(nn.Sequential):
    """ Parallel container that runs each module on the input and combines the different outputs by concatenating them.
    The tensors are considered as batched tensors and are thus concatenated in dimension 1.

    Args:
        *args: Arguments passed to :class:`~lightnet.network.layer._util.Parallel`
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
        output = torch.cat([module(x) for name, module in self.named_children() if name != 'post'], dim=1)
        if self.post is not None:
            output = self.post(output)

        return output


class ParallelSum(nn.Sequential):
    """ Parallel container that runs each module on the input and combines the different outputs by summing them.

    Args:
        *args: Arguments passed to :class:`~lightnet.network.layer._util.Parallel`
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
        output = torch.sum(torch.stack([module(x) for name, module in self.named_children() if name != 'post']), dim=0)
        if self.post is not None:
            output = self.post(output)

        return output
