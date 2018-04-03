#
#   Base lightnet network structure
#   Copyright EAVISE
#

import os
import collections
import logging
import torch
import torch.nn as nn

from .weight import *

__all__ = ['Darknet']
log = logging.getLogger(__name__)

class Darknet(nn.Module):
    """ This class provides an abstraction layer on top of the ``pytorch Module``
    to make it easier to implement the darknet networks. There are 2 basic ways of using this class:

    - Override the ``forward()`` function.
      This is the easiest solution for people who already know how to use pytorch.
      This module then only adds the benefit of being able to load in darknet weights.
    - Define ``self.loss`` and ``self.postprocess`` as functions and override the ``_forward()`` function.
      This class will then automatically call the loss and postprocess functions on the output of ``_forward()``,
      depending whether the network is training or evaluating.

    Attributes:
        self.seen (int): The number of images the network has processed to train (used by engine)
        self.input_dim (list): Input dimensions of the network (used by data transforms)

    Note:
        If you define **self.layers** as a :class:`pytorch:torch.nn.Sequential` or :class:`pytorch:torch.nn.ModuleList`,
        the default ``_forward()`` function can use these layers automatically to run the network.

    Warning:
        If you use your own ``forward()`` function, you need to update the **self.seen** parameter
        whenever the network is training.
    """
    def __init__(self):
        super(Darknet, self).__init__()

        # Parameters
        self.layers = None
        self.loss = None
        self.postprocess = None
        self.header = [0,2,0]
        self.seen = 0

    def _forward(self, x):
        log.debug('Running default forward functions')
        if isinstance(self.layers, nn.Sequential):
            return self.layers(x)
        elif isinstance(self.layers, nn.ModuleList):
            log.warn('No _forward function defined, looping sequentially over modulelist')
            for _,module in enumerate(self.layers):
                x = module(x)
            return x
        else:
            log.error(f'No _forward function defined and no default behaviour for this type of layers [{type(self.layers)}]')
            raise NotImplementedError

    def forward(self, x, target=None):
        """ This default forward function will compute the output of the network as ``self._forward(x)``.
        Then, depending on whether you are training or evaluating, it will pass that output to ``self.loss()`` or ``self.posprocess()``. |br|
        This function also increments the **self.seen** variable.

        Args:
            x (torch.autograd.Variable): Input variable
            target (torch.autograd.Variable, optional): Target for the loss function; Required if training and optional otherwise (see note)

        Note:
            If you are evaluating your network and you pass a target variable, the network will return a (output, loss) tuple.
            This is usefull for testing your network, as you usually want to know the validation loss.
        """
        if self.training:
            self.seen += x.size(0)
            x = self._forward(x)

            if callable(self.loss):
                return self.loss(x, target)
            else:
                return x
        else:
            x = self._forward(x)

            if target is not None and callable(self.loss):
                loss = self.loss(x.clone(), target)
            else:
                loss = None

            if callable(self.postprocess):
                x = self.postprocess(x)

            if loss is not None:
                return x, loss
            else:
                return x

    def modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.

        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod == None:
            mod = self

        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                yield from self.modules_recurse(module)
            else:
                yield module

    def load_weights(self, weights_file):
        """ This function will load the weights from a file.
        If the file extension is ``.pt``, it will be considered as a `pytorch pickle file <http://pytorch.org/docs/0.3.0/notes/serialization.html#recommended-approach-for-saving-a-model>`_. 
        Otherwise, the file is considered to be a darknet binary weight file.

        Args:
            weights_file (str): path to file
        """
        if weights_file is not None:
            if os.path.splitext(weights_file)[1] == '.pt':
                log.info('Loading weights from pytorch file')
                self._load_pickle_weights(weights_file)
            else:
                log.info('Loading weights from darknet file')
                self._load_darknet_weights(weights_file)

            if hasattr(self.loss, 'seen'):
                self.loss.seen = self.seen

    def save_weights(self, weights_file):
        """ This function will save the weights to a file.
        If the file extension is ``.pt``, it will be considered as a `pytorch pickle file <http://pytorch.org/docs/0.3.0/notes/serialization.html#recommended-approach-for-saving-a-model>`_. 
        Otherwise, the file is considered to be a darknet binary weight file.

        Args:
            weights_file (str): path to file
        """
        if weights_file is not None:
            if os.path.splitext(weights_file)[1] == '.pt':
                log.debug('Saving weights to pytorch file')
                self._save_pickle_weights(weights_file)
            else:
                log.debug('Saving weights to darknet file')
                self._save_darknet_weights(weights_file)

    def update_weights(self, weights_file):
        """ Pytorch weight files does not allow for partial loading of a network.
        This update function gets around it by updating the current state_dict of the network
        with the state_dict of the pytorch ``weights_file`` given.
        """
        old_state = self.state_dict()
        new_state = torch.load(weights_file, lambda storage, loc: storage)

        # Changed in layer.py: self.layer -> self.layers
        for key in list(new_state['weights'].keys()):
            if '.layer.' in key:
                log.deprecated('Deprecated weights file found. Consider resaving your weights file before this manual intervention gets removed')
                new_key = key.replace('.layer.', '.layers.')
                new_state['weights'][new_key] = new_state['weights'].pop(key)

        new_dict = {k: v for k,v in new_state['weights'].items() if k in old_state}
        old_state.update(new_dict)
        self.load_state_dict(old_state)

        self.seen = new_state['seen']
        if hasattr(self.loss, 'seen'):
            self.loss.seen = self.seen

    def _load_darknet_weights(self, weights_file):
        weights = WeightLoader(weights_file)
        self.header = weights.header
        self.seen = weights.seen

        for module in self.modules_recurse():
            try:
                weights.load_layer(module)
                log.info(f'Layer loaded: {module}')
                if weights.start >= weights.size:
                    log.debug(f'Finished loading weights [{weights.start}/{weights.size} weights]')
                    break
            except NotImplementedError:
                log.info(f'Layer skipped: {module.__class__.__name__}')

    def _save_darknet_weights(self, weights_file):
        weights = WeightSaver(self.header, self.seen)

        for module in self.modules_recurse():
            try:
                weights.save_layer(module)
                log.info(f'Layer saved: {module}')
            except NotImplementedError:
                log.info(f'Layer skipped: {module.__class__.__name__}')

        weights.write_file(weights_file)

    def _load_pickle_weights(self, weights_file):
        state = torch.load(weights_file, lambda storage, loc: storage)
        self.seen = state['seen']

        # Changed in layer.py: self.layer -> self.layers
        for key in list(state['weights'].keys()):
            if '.layer.' in key:
                log.deprecated('Deprecated weights file found. Consider resaving your weights file before this manual intervention gets removed')
                new_key = key.replace('.layer.', '.layers.')
                state['weights'][new_key] = state['weights'].pop(key)

        self.load_state_dict(state['weights'])

    def _save_pickle_weights(self, weights_file):
        state = {
            'seen': self.seen,
            'weights': self.state_dict()
        }
        torch.save(state, weights_file)
        log.info(f'Weight file saved as {weights_file}')
