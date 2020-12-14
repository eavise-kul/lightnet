#
#   Base lightnet network module structure
#   Copyright EAVISE
#

import inspect
import logging
import re
from collections import OrderedDict
import torch
import torch.nn as nn

__all__ = ['Lightnet']
log = logging.getLogger(__name__)


class Lightnet(nn.Module):
    """ This class provides an abstraction layer on top of :class:`pytorch:torch.nn.Module` and is used as a base for every network implemented in this framework.

    Note:
        If you define **self.layers** as a :class:`pytorch:torch.nn.Sequential` or :class:`pytorch:torch.nn.ModuleList`,
        the default ``forward()`` function can use these layers automatically to run the network.
    """
    def __init__(self):
        super().__init__()
        self.layers = None

    def forward(self, x):
        log.debug('Running default forward function')
        if hasattr(self, 'layers'):
            if isinstance(self.layers, nn.Sequential):
                return self.layers(x)
            elif isinstance(self.layers, nn.ModuleList):
                log.warning('No _forward function defined, looping sequentially over modulelist')
                for _, module in enumerate(self.layers):
                    x = module(x)
                return x
            else:
                raise NotImplementedError(f'No _forward function defined and no default behaviour for this type of layers [{type(self.layers)}]')
        else:
            raise NotImplementedError(f'No _forward function defined and no default behaviour for this network')

    def layer_loop(self, mod=None):
        """ This function will recursively loop over all moduleList and Sequential children.

        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**

        Returns:
            (generator): Iterator that will loop over and yield the different layers
        """
        if mod is None:
            mod = self

        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                yield from self.layer_loop(module)
            else:
                yield module

    def named_layer_loop(self, mod=None):
        """ Named version of :func:`~lightnet.network.module.Lightnet.layer_loop`

        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**

        Returns:
            (generator): Iterator that will loop over and yield (name, layer) tuples
        """
        if mod is None:
            mod = self

        for name, module in mod.named_children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                yield from self.named_layer_loop(module)
            else:
                yield name, module

    def load(self, weights_file, remap=None, strict=True):
        """ This function will load the weights from a file.
        It also allows to load in a weights file with only a part of the weights in.

        Args:
            weights_file (str): path to file
            remap (callable or list, optional): Remapping of the weights, see :func:`~lightnet.network.module.Lightnet.weight_remapping`; Default **None**
            strict (Boolean, optional): Whether the weight file should contain all layers of the model; Default **True**

        Note:
            This function will load the weights to CPU,
            so you should use ``network.to(device)`` afterwards to send it to the device of your choice.
        """
        keys = self.state_dict().keys()

        if remap is not None:
            state = self.weight_remapping(torch.load(weights_file, 'cpu'), remap)
            remap = ' remapped'
        else:
            state = torch.load(weights_file, 'cpu')
            remap = ''

        log.info(f'Loading{remap} weights from file [{weights_file}]')
        if not strict and state.keys() != keys:
            log.warning('Modules not matching, performing partial update')

        self.load_state_dict(state, strict=strict)

    def load_pruned(self, weights_file, strict=True):
        """ This function will load pruned weights from a file.
        It also allows to load a weights file,
        which contains less channels in a convolution than orginally defined in the network.

        Args:
            weights_file (str): path to file
            strict (Boolean, optional): Whether the weight file should contain all layers of the model; Default **True**

        Note:
            This function will load the weights to CPU,
            so you should use ``network.to(device)`` afterwards to send it to the device of your choice.
        """
        keys = set(self.state_dict().keys())
        log.info(f'Loading pruned weights from file [{weights_file}]')
        state = torch.load(weights_file, 'cpu')

        # Prune tensors
        for key, val in state.items():
            if key in keys:
                tensor = self
                module = self
                name = None
                for p in key.split("."):
                    module = tensor
                    name = p
                    tensor = getattr(tensor, p)

                if tensor.shape != val.shape:
                    slices = [slice(0, s) for s in val.shape]
                    if isinstance(tensor, torch.nn.Parameter):
                        setattr(module, p, torch.nn.Parameter(tensor[slices]))
                    else:
                        setattr(module, p, tensor[slices])

        # Modify module metadata
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                if module.groups == 1:
                    module.in_channels = module.weight.shape[1]
                    module.out_channels = module.weight.shape[0]
                elif module.groups == module.in_channels == module.out_channels:
                    module.out_channels = module.weight.shape[0]
                    module.in_channels = module.out_channels
                    module.groups = module.out_channels
            elif isinstance(module, torch.nn.BatchNorm2d):
                if module.weight is not None:
                    module.num_features = module.weight.shape[0]
                elif module.running_mean is not None:
                    module.num_features = module.running_mean.shape[0]

        # Load weights
        if not strict and state.keys() != keys:
            log.warning('Modules not matching, performing partial update')
        self.load_state_dict(state, strict=strict)

    def save(self, weights_file, remap=None):
        """ This function will save the weights to a file.

        Args:
            weights_file (str): path to file
            remap (callable or list, optional): Remapping of the weights, see :func:`~lightnet.network.module.Lightnet.weight_remapping`; Default **None**
        """
        if remap is not None:
            state = self.weight_remapping(self.state_dict(), remap)
            remap = ' remapped'
        else:
            state = self.state_dict()
            remap = ''

        torch.save(state, weights_file)
        log.info(f'Saved{remap} weights as {weights_file}')

    def __str__(self):
        """ Shorter version than default PyTorch one. """
        args = list(inspect.signature(self.__class__.__init__).parameters.keys())
        args.remove('self')

        string = self.__class__.__module__ + '.' + self.__class__.__name__ + '('
        for i, arg in enumerate(args):
            if i > 0:
                string += ', '
            val = getattr(self, arg, '?')
            string += f'{arg}={val}'
        string += ')'

        return string

    @staticmethod
    def weight_remapping(weights, remap):
        """ This function is used to remap the keys of a ``state_dict``.
        This can be usefull to load in weights from a different framework, or to modify weights from a backbone network, for usage in another (detection) network. |br|
        This method does not usually get called directly, but is used by :func:`~lightnet.network.module.Lightnet.load` and :func:`~lightnet.network.module.Lightnet.save`
        to modify the weights prior to loading/saving them.

        Args:
            weights (dict): The weights state dictionary
            remap (callable or list): Remapping of the weights, see Note

        Note:
            The optional ``remap`` parameter expects a callable object or a list of tuples.

            If the argument is callable, it will be called with each key in the ``state_dict`` and it should return a new key.
            If it returns **None**, that weight will be removed from the new state_dict.

            if the ``remap`` argument is a list of tuples, they should contain **('old', 'new')** remapping sequences.
            The remapping sequence can contain strings or regex objects. |br|
            What happens when you supply a remapping list,
            is that this function will loop over the ``state_dict`` of the model and for each parameter of the ``state_dict`` it will loop through the remapping list.
            If the first string or regex of the remapping sequence is found in the ``state_dict`` key, it will be replaced by the second string or regex of that remapping sequence. |br|
            There are two important things to note here:

            - If a key does not match any remapping sequence, it gets discarded. To save all the weights, even if you need no remapping, add a last remapping sequence of **('', '')** which will match with all keys, but not modify them.
            - The remapping sequences or processed in order. This means that if a key matches with a certain remapping sequence, the following sequences will not be considered anymore.
        """
        new_weights = OrderedDict()

        if callable(remap):
            for k, v in weights.items():
                nk = remap(k)
                if nk is not None:
                    new_weights[nk] = v
        else:
            for k, v in weights.items():
                for r in remap:
                    if re.match(r[0], k) is not None:
                        new_weights[re.sub(r[0], r[1], k)] = v
                        break

        return new_weights
