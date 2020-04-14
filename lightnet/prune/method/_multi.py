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
    """ TODO : docstring

    Note:
        Runs pruners in the order they were declared upon initialisation.

    Note:
        Skips a pruner if percentage is 0 (or less).
    """
    def __init__(self, *pruners):
        self.pruners = flatten(pruners)

    def __call__(self, *percentages):
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
            Since the number of prunable channels is independent of the pruning technique,
            we just return the property from the first pruner in the list.

        Returns:
            int: Number of prunable channels of the entire model
        """
        return self.pruners[0].prunable_channels

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
