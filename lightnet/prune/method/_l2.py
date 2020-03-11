#
#   Normalized L2 pruning
#   Copyright EAVISE
#

from functools import reduce
import itertools
import logging
import math
import torch
from ._base import *

__all__ = ['L2Pruner']
log = logging.getLogger(__name__)


class L2Pruner(Pruner):
    """ TODO

    Normalized L2 : https://arxiv.org/pdf/1611.06440.pdf -> 2.3 Normalization

    Warning:
        Percentage is approximative. |br|
        If L2 tells to prune all filters in a convolution,
        we keep the most important and reduce the number of pruned filters by one.
    """
    def prune(self, percentage, prune_manner):
        # Get normalized L2
        l2_norms = []
        num_features = []
        for key, dependency in self.dependencies.items():
            w = dependency.module.weight.view(dependency.module.out_channels, -1)
            l2 = w.pow(2).sum(1) / w.shape[1]
            l2 /= torch.sqrt(l2.pow(2).sum())

            l2_norms.append(l2)
            num_features.append((key, l2.shape[0]))

        # Select filters to prune
        indices = torch.argsort(torch.cat(l2_norms))
        num_pruned = math.floor(indices.numel() * percentage)
        if num_pruned <= 0:
            return num_pruned

        # Convert filters total index to index per convolution
        indices = indices[:num_pruned].tolist()
        indices = map(get_idx_per_conv, indices, itertools.repeat(num_features))
        indices = reduce(get_filter_dict, indices, dict())

        # Prune
        num_features = dict(num_features)
        for key, filter_list in indices.items():
            num = num_features[key]
            while len(filter_list) >= num:
                filter_list = filter_list[:-1]
                num_pruned -= 1

            if len(filter_list) > 0:
                prune_manner(self.dependencies[key], filter_list)

        return num_pruned


def get_idx_per_conv(idx, num_features):
    for name, length in num_features:
        if idx >= length:
            idx -= length
        else:
            return (name, idx)


def get_filter_dict(filter_dict, idx):
    name, idx = idx

    if name not in filter_dict:
        filter_dict[name] = [idx]
    else:
        filter_dict[name].append(idx)

    return filter_dict
