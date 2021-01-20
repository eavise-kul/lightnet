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

    .. figure:: /.static/api/residual.*
       :width: 100%
       :alt: Residual module design

    Note:
        If you are using an OrderedDict to pass the modules to the sequential,
        you can set the `skip` and `post` values inside of that dict as well.

    Example:
        >>> # Note that the input channels and output channels should be the same for each branch (ic. 3 and 32)
        >>> module = ln.network.layer.Residual(
        ...     ln.network.layer.Conv2dBatchReLU(3, 32, 3, 1, 1),
        ...     ln.network.layer.Conv2dBatchReLU(32, 64, 3, 1, 1),
        ...     ln.network.layer.Conv2dBatchReLU(64, 32, 3, 1, 1),
        ...     # The skip block should ensure that the output tensor has the same number of channels
        ...     skip=ln.network.layer.Conv2dBatchReLU(3, 32, 1, 1, 0),
        ...     # The post block should run on the summed tensor,
        ...     # so the in_channels are equal to the out_channels of the output of the residual
        ...     post=ln.network.layer.Conv2dBatchReLU(32, 1, 1, 1, 0)
        ... )
        >>> print(module)
        Residual(
          (0): Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          (1): Conv2dBatchReLU(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          (2): Conv2dBatchReLU(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          (skip): Conv2dBatchReLU(3, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), ReLU(inplace=True))
          (post): Conv2dBatchReLU(32, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), ReLU(inplace=True))
        )
        >>> in_tensor = torch.rand(1, 3, 10, 10)
        >>> out_tensor = module(in_tensor)
        >>> out_tensor.shape
        torch.Size([1, 1, 10, 10])
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
            return_selection (bool): Whether to return the selected layers, or just store them as `self.selected`
            *args: Arguments that are passed to the Sequential init function

        .. figure:: /.static/api/sequentialselect.*
           :width: 100%
           :alt: SequentialSelect module design

        Note:
            If your selection only contains one item, this layer will flatten the return output tensor:

            >>> main_output, selected_output = layer(input)  # doctest: +SKIP

            However, if you have multiple outputs in your selection list, the outputs will be grouped in a dictionary:

            >>> main_output, selected_output_dict = layer(input)  # doctest: +SKIP

            The same applies for `self.selected` in case `return_selection` is set to False.

        Example:
            >>> module = ln.network.layer.SequentialSelect(
            ...     # We want to return the output from layers '1' and '3'
            ...     [1, 3],
            ...
            ...     # Since we specify False, the selected outputs will not be returned,
            ...     # but we can access them as `module.selected`
            ...     False,
            ...
            ...     # Sequential
            ...     ln.network.layer.Conv2dBatchReLU(3, 32, 3, 1, 1),
            ...     ln.network.layer.Conv2dBatchReLU(32, 64, 3, 1, 1),
            ...     ln.network.layer.Conv2dBatchReLU(64, 64, 3, 1, 1),
            ...     ln.network.layer.Conv2dBatchReLU(64, 32, 3, 1, 1),
            ...     ln.network.layer.Conv2dBatchReLU(32, 64, 3, 1, 1),
            ...     ln.network.layer.Conv2dBatchReLU(64, 128, 3, 1, 1),
            ... )
            >>> print(module)
            SequentialSelect(
              selection=['1', '3'], return=False, flatten=False
              (0): Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
              (1): Conv2dBatchReLU(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
              (2): Conv2dBatchReLU(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
              (3): Conv2dBatchReLU(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
              (4): Conv2dBatchReLU(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
              (5): Conv2dBatchReLU(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            )
            >>> in_tensor = torch.rand(1, 3, 10, 10)
            >>> out_tensor = module(in_tensor)
            >>> print(out_tensor.shape)
            torch.Size([1, 128, 10, 10])
            >>> print(module.selected['1'].shape)
            torch.Size([1, 64, 10, 10])
            >>> print(module.selected['3'].shape)
            torch.Size([1, 32, 10, 10])

            >>> # Setting return_selection to True means the module will return a tuple of (output, selection)
            >>> module = ln.network.layer.SequentialSelect(
            ...     [1, 3], True,
            ...     ln.network.layer.Conv2dBatchReLU(3, 32, 3, 1, 1),
            ...     ln.network.layer.Conv2dBatchReLU(32, 64, 3, 1, 1),
            ...     ln.network.layer.Conv2dBatchReLU(64, 64, 3, 1, 1),
            ...     ln.network.layer.Conv2dBatchReLU(64, 32, 3, 1, 1),
            ...     ln.network.layer.Conv2dBatchReLU(32, 64, 3, 1, 1),
            ...     ln.network.layer.Conv2dBatchReLU(64, 128, 3, 1, 1),
            ... )
            >>> in_tensor = torch.rand(1, 3, 10, 10)
            >>> out_tensor, selected = module(in_tensor)
            >>> print(out_tensor.shape)
            torch.Size([1, 128, 10, 10])
            >>> print(selected['1'].shape)
            torch.Size([1, 64, 10, 10])
            >>> print(selected['3'].shape)
            torch.Size([1, 32, 10, 10])
    """
    def __init__(self, selection, return_selection, *args):
        super().__init__(*args)

        self.return_selection = return_selection
        self.selected = None
        self.selection = [str(select) for select in selection]
        self.flatten = len(self.selection) == 1
        k = list(self._modules.keys())
        for sel in self.selection:
            if sel not in k:
                raise KeyError(f'Selection key not found in sequential [{sel}]')

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

    .. figure:: /.static/api/parallel.*
       :width: 100%
       :alt: Parallel module design

    Example:
        >>> # Note that the input channels should be the same for each branch (ic. 3)
        >>> module = ln.network.layer.Parallel(
        ...     ln.network.layer.Conv2dBatchReLU(3, 16, 3, 1, 1),
        ...     torch.nn.Sequential(
        ...         ln.network.layer.Conv2dBatchReLU(3, 8, 3, 1, 1),
        ...         ln.network.layer.Conv2dBatchReLU(8, 16, 3, 1, 1),
        ...         ln.network.layer.Conv2dBatchReLU(16, 32, 3, 1, 1),
        ...     ),
        ...     ln.network.layer.InvertedBottleneck(3, 64, 3, 1, 1),
        ... )
        >>> print(module)
        Parallel(
          (0): Conv2dBatchReLU(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          (1): Sequential(
            (0): Conv2dBatchReLU(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (1): Conv2dBatchReLU(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (2): Conv2dBatchReLU(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          )
          (2): InvertedBottleneck(3, 64, kernel_size=(3, 3), stride=(1, 1), expansion=1, ReLU(inplace=True))
        )
        >>>
        >>> in_tensor = torch.rand(1, 3, 10, 10)
        >>> out1, out2, out3 = module(in_tensor)
        >>> out1.shape
        torch.Size([1, 16, 10, 10])
        >>> out2.shape
        torch.Size([1, 32, 10, 10])
        >>> out3.shape
        torch.Size([1, 64, 10, 10])
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

    .. figure:: /.static/api/parallelcat.*
       :width: 100%
       :alt: ParallelCat module design

    Note:
        If you are using an `OrderedDict` to pass the modules to the sequential,
        you can set the `post` value inside of that dict as well.

    Example:
        >>> # Note that the input channels should be the same for each branch (ic. 3)
        >>> module = ln.network.layer.ParallelCat(
        ...     ln.network.layer.Conv2dBatchReLU(3, 32, 3, 1, 1),
        ...     torch.nn.Sequential(
        ...         ln.network.layer.Conv2dBatchReLU(3, 32, 3, 1, 1),
        ...         ln.network.layer.Conv2dBatchReLU(32, 64, 3, 1, 1),
        ...         ln.network.layer.Conv2dBatchReLU(64, 32, 3, 1, 1),
        ...     ),
        ...     ln.network.layer.InvertedBottleneck(3, 32, 3, 1, 1),
        ...
        ...     # The post block should run on the concatenated tensor,
        ...     # so the in_channels are equal to the sum of the out_channels of the parallel modules
        ...     post=ln.network.layer.Conv2dBatchReLU(96, 32, 1, 1, 0)
        ... )
        >>> print(module)
        ParallelCat(
          (0): Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          (1): Sequential(
            (0): Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (1): Conv2dBatchReLU(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (2): Conv2dBatchReLU(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          )
          (2): InvertedBottleneck(3, 32, kernel_size=(3, 3), stride=(1, 1), expansion=1, ReLU(inplace=True))
          (post): Conv2dBatchReLU(96, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), ReLU(inplace=True))
        )
        >>> in_tensor = torch.rand(1, 3, 10, 10)
        >>> out_tensor = module(in_tensor)
        >>> out_tensor.shape
        torch.Size([1, 32, 10, 10])
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

    .. figure:: /.static/api/parallelsum.*
       :width: 100%
       :alt: ParallelSum module design

    Note:
        If you are using an `OrderedDict` to pass the modules to the sequential,
        you can set the `post` value inside of that dict as well.

    Example:
        >>> # Note that the input channels and output channels should be the same for each branch (ic. 3 and 32)
        >>> module = ln.network.layer.ParallelSum(
        ...     ln.network.layer.Conv2dBatchReLU(3, 32, 3, 1, 1),
        ...     torch.nn.Sequential(
        ...         ln.network.layer.Conv2dBatchReLU(3, 32, 3, 1, 1),
        ...         ln.network.layer.Conv2dBatchReLU(32, 64, 3, 1, 1),
        ...         ln.network.layer.Conv2dBatchReLU(64, 32, 3, 1, 1),
        ...     ),
        ...     ln.network.layer.InvertedBottleneck(3, 32, 3, 1, 1),
        ...
        ...     # The post block should run on the summed tensor,
        ...     # so the in_channels are equal to the out_channels of the parallel modules
        ...     post=ln.network.layer.Conv2dBatchReLU(32, 1, 1, 1, 0),
        ... )
        >>> print(module)
        ParallelSum(
          (0): Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          (1): Sequential(
            (0): Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (1): Conv2dBatchReLU(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
            (2): Conv2dBatchReLU(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ReLU(inplace=True))
          )
          (2): InvertedBottleneck(3, 32, kernel_size=(3, 3), stride=(1, 1), expansion=1, ReLU(inplace=True))
          (post): Conv2dBatchReLU(32, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), ReLU(inplace=True))
        )
        >>> in_tensor = torch.rand(1, 3, 10, 10)
        >>> out_tensor = module(in_tensor)
        >>> out_tensor.shape
        torch.Size([1, 1, 10, 10])
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
