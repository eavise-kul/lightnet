#
#   Base darknet network structure
#   Copyright EAVISE
#

import os
import collections
import random

import numpy as np
import torch
import torch.nn as nn

from .logger import *
from .weights import *

__all__ = ['Darknet']

class Darknet(nn.Module):
    """ Base network class to create darknet CNNs """
    def __init__(self):
        super(Darknet, self).__init__()

        # Parameters
        self.layers = None
        self.loss = None
        self.postprocess = None
        self.header = [0,2,0]
        self.seen = 0
        self.input_dim = [0,0,0]
        self.num_classes = 0
        self.anchors = []
        self.num_anchors = 0

    def _forward(self, x):
        log(Loglvl.VERBOSE, 'Running default forward functions')
        if isinstance(self.layers, nn.Sequential):
            return self.layers(x)
        elif isinstance(self.layers, nn.ModuleList):
            log(Loglvl.WARN, 'No _forward function defined, looping sequentially over modulelist')
            for _,module in enumerate(self.layers):
                x = module(x)
            return x
        else:
            log(Loglvl.Error, f'No _forward function defined and no default behaviour for this type of layers [{type(self.layers)}]', NotImplementedError)

    def forward(self, x, target=None):
        x = self._forward(x)

        if self.training and callable(self.loss):
            return self.loss(x, target)
        elif not self.training and callable(self.postprocess):
            if target is not None and callable(self.loss):
                loss = self.loss(x.clone(), target)
                return self.postprocess(x), loss
            else:
                return self.postprocess(x)
        else:
            return x

    def modules_recurse(self, mod=None):
        if mod == None:
            mod = self

        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                yield from self.modules_recurse(module)
            else:
                yield module

    def change_input_dim(self, multiple=32):
        """ Change input dimensions for training """
        size = (random.randint(0,9) + 10) * multiple 
        log(Loglvl.VERBOSE, f'Resizing network [{size}]')

        if not self.training:
            log(Loglvl.WARN, 'Changing input dimensions whilst not training')

        self.input_dim[:2] = [size, size]

    def load_weights(self, weights_file):
        if weights_file is not None:
            if os.path.splitext(weights_file)[1] == '.pt':
                log(Loglvl.VERBOSE, 'Loading weights from pytorch file')
                self._load_pickle_weights(weights_file)
            else:
                log(Loglvl.VERBOSE, 'Loading weights from darknet file')
                self._load_darknet_weights(weights_file)

    def save_weights(self, weights_file):
        if weights_file is not None:
            if os.path.splitext(weights_file)[1] == '.pt':
                log(Loglvl.DEBUG, 'Saving weights to pytorch file')
                self._save_pickle_weights(weights_file)
            else:
                log(Loglvl.DEBUG, 'Saving weights to darknet file')
                self._save_darknet_weights(weights_file)

    def _load_darknet_weights(self, weights_file):
        weights = WeightLoader(weights_file)
        self.header = weights.header
        self.seen = weights.seen

        for module in self.modules_recurse():
            try:
                weights.load_layer(module)
                log(Loglvl.VERBOSE, f'Layer loaded: {module}')
                if weights.start >= weights.size:
                    log(Loglvl.DEBUG, f'Finished loading weights [{weights.start}/{weights.size} weights]')
                    break
            except NotImplementedError:
                log(Loglvl.VERBOSE, f'Layer skipped: {module.__class__.__name__}')

    def _save_darknet_weights(self, weights_file):
        weights = WeightSaver(self.header, self.seen)

        for module in self.modules_recurse():
            try:
                weights.save_layer(module)
                log(Loglvl.VERBOSE, f'Layer saved: {module}')
            except NotImplementedError:
                log(Loglvl.VERBOSE, f'Layer skipped: {module.__class__.__name__}')

        weights.write_file(weights_file)

    def _load_pickle_weights(self, weights_file):
        state = torch.load(weights_file, lambda storage, loc: storage)
        self.seen = state['seen']
        self.load_state_dict(state['weights'])

    def _save_pickle_weights(self, weights_file):
        state = {
            'seen': self.seen,
            'weights': self.state_dict()
        }
        torch.save(state, weights_file)
        log(Loglvl.VERBOSE, f'Weight file saved as {weights_file}')
