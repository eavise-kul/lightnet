#
#   Learning Rate Scheduling
#   Copyright EAVISE
#

import torch
import logging

__all__ = ['SchedulerCompositor']
log = logging.getLogger(__name__)


class SchedulerCompositor:
    def __init__(self, *args):
        if len(args) == 0:
            raise ValueError('Compositor requires at least one scheduler')

        self.counts, self.sched = zip(*args)
        if not all(c1 < c2 for c1, c2 in zip(self.counts, self.counts[1:])):
            raise ValueError('Count values need to be strictly increasing')

    def step(self, count, **kwargs):
        for i, c in enumerate(self.counts):
            if count < c:
                i -= 1
                break

        if i < 0:
            log.error(f'No Scheduler defined for count value of {count}')
            return

        return self.sched[i].step(**kwargs)

    def state_dict(self):
        state_dict = [s.state_dict() for s in self.sched]

        # Dont save lambdas/functions
        # -> it is pointless as you need the definitions before unpickling, so you have the functions already
        # TODO: Remove this quickfix once https://github.com/pytorch/pytorch/pull/9927 lands in a release
        for sd in state_dict:
            if 'lr_lambdas' in sd:
                del sd['lr_lambdas']

        return state_dict

    def load_state_dict(self, state):
        [self.sched[i].load_state_dict(s) for i, s in enumerate(state)]
