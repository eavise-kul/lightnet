import logging
import math
import torch
from ._base import *

__all__ = ['GeometricMedianPruner']
log = logging.getLogger(__name__)


class GeometricMedianPruner(Pruner):
    """ TODO

    FPGM : https://arxiv.org/pdf/1811.00250.pdf

    Warning:
        The percentage given is used on a per-layer basis for this pruning method.
        This means that if you give a percentage of 10%, we prune 10% of each prunable layer.

    Note:
        The percentage is approximative. |br|
        If the pruner tells to prune all filters in a convolution,
        we keep the most important and reduce the number of pruned filters by one.
    """
    def __init__(self, *args, p_norm=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_norm = p_norm

    def prune(self, percentage, prune_manner):
        total_pruned = 0

        for key, dependency in self.dependencies.items():
            # Compute number of filters to prune
            num_pruned = math.floor(percentage * dependency.module.out_channels)
            if num_pruned >= dependency.module.out_channels:
                num_pruned = dependency.module.out_channels - 1
            if num_pruned <= 0:
                continue
            total_pruned += num_pruned

            # Get pairwise distance between all filters and sum them to get similarity score per filter
            w = dependency.module.weight.view(dependency.module.out_channels, -1)
            similarity = torch.cdist(w, w, p=self.p_norm).abs().sum(dim=0)

            # Prune selected filters
            indices = torch.argsort(similarity)[:num_pruned].tolist()
            prune_manner(dependency, indices)

        return total_pruned
