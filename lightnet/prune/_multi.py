#
#   Combine multiple pruners
#   Copyright EAVISE
#
import collections
import logging

__all__ = ['MultiPruner']
log = logging.getLogger(__name__)


def flatten(x):
    """ Only works with lists of non-iterable data (eg. not strings) """
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


class MultiPruner:
    """ Combines different pruner methods together. |br|
    This class takes a list of pruners and applies each of them in order.
    It can be used to simultaneously prune different networks or different parts of a network,
    or even run different pruners on a single network.

    Args:
        *pruners (list <lightnet.prune.Pruner>): List of pruners to run

    Note:
        The MultiPruner runs pruners in the order they were declared upon initialisation.
    """
    def __init__(self, *pruners):
        self.pruners = flatten(pruners)

    def __call__(self, *percentages):
        """ Actually perform the pruning.

        Args:
            *percentages (list <float>): percentage for each pruner

        Note:
            If the percentage for a certain pruner is zero, that pruner gets skipped.
        """
        percentages = flatten(percentages)
        if len(percentages) < len(self.pruners):
            raise ValueError(f'Received {len(percentages)} percentages, but we have {len(self.pruners)} pruners.')
        elif len(percentages) > len(self.pruners):
            log.error(f'Received {len(percentages)} percentages, but we have {len(self.pruners)} pruners.')

        total_pruned = 0
        for pruner, percentage in zip(self.pruners, percentages):
            if percentage <= 0:
                continue
            total_pruned += pruner(percentage)

        return total_pruned

    @property
    def prunable_channels(self):
        """ Returns the number of prunable channels left on the module. |br|
        This property is added, so that you can use a :class:`~lightnet.prune.MultiPruner`
        anywhere you would use a regular :class:`~lightnet.prune.Pruner`,
        but it does not make a whole lot of sense for this pruner, as each individual pruner could have a different model.

        We thuse simply return the ``prunable_channels`` property from the first pruner.

        Returns:
            int: Number of prunable channels of the entire model
        """
        return self.pruners[0].prunable_channels

    @property
    def pruned_channels(self):
        """ Returns the number of pruned channels from the last pruning operation. """
        return sum(p.pruned_channels for p in self.pruners)

    @property
    def soft_pruned_channels(self):
        """ Returns the number of soft pruned channels from the last pruning operation. """
        return sum(p.soft_pruned_channels for p in self.pruners)

    @property
    def hard_pruned_channels(self):
        """ Returns the number of hard pruned channels from the last pruning operation. """
        return sum(p.hard_pruned_channels for p in self.pruners)

    def __str__(self):
        string = f'{self.__class__.__name__} ['
        for pruner in self.pruners:
            string += f'{str(pruner)}, '
        return string[:-2] + ']'

    def __repr__(self):
        string = f'{self.__class__.__name__} [\n'
        for pruner in self.pruners:
            prunerrepr = repr(pruner)
            if '\n' in prunerrepr:
                prunerrepr = prunerrepr.replace('\n', '\n    ')
            string += f'  {prunerrepr},\n'
        return string + ']'
