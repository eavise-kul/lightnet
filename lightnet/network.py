#
#   Base darknet network structure
#   Copyright EAVISE
#

import collections

import numpy as np
import torch
import torch.nn as nn

from .logger import *
from .weights import *

class Darknet(nn.Module):
    """ Base network class to create darknet CNNs """
    def __init__(self):
        super(Darknet, self).__init__()

        self.layers = None
        self.loss = None
        self.postprocess = None
        self.header = [0,2,0]
        self.seen = 0

    def _forward(self, x):
        log(Loglvl.DEBUG, 'Running default forward functions')
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

    def load_darknet_weights(self, weights_file):
        weights = WeightLoader(weights_file)
        self.header = weights.header
        self.seen = weights.seen

        for module in self.modules_recurse():
            try:
                weights.load_layer(module)
                log(Loglvl.DEBUG, f'Layer Loaded: {module}')
                if weights.start >= weights.size:
                    log(Loglvl.DEBUG, f'Finished loading weights [{weights.start}/{weights.size} weights]')
            except NotImplementedError:
                log(Loglvl.DEBUG, f'Layer skipped: {module.__class__.__name__}')

    def save_darknet_weights(self, weights_file):
        weights = WeightSaver(self.header, self.seen)

        for module in self.modules_recurse():
            try:
                weights.save_layer(module)
                log(Loglvl.DEBUG, f'Layer Saved: {module}')
            except NotImplementedError:
                log(Loglvl.DEBUG, f'Layer skipped: {module.__class__.__name__}')

        weights.write_file(weights_file)

    def load_pickle_weights(self, weights_file):
        log(Loglvl.ERROR, 'This is not yet implemented, please use darknet format', NotImplementedError)
        state = torch.load(weights_file)
        self.seen = state['seen']
        self.load_state_dict(state['weights'])

    def save_pickle_weights(self, weights_file):
        log(Loglvl.ERROR, 'This is not yet implemented, please use darknet format', NotImplementedError)
        state = {
            'seen': self.seen,
            'weights': self.state_dict(),
        }
        torch.save(state, weights_file)
