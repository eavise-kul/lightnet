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
        return [s.state_dict() for s in self.sched]

    def load_state_dict(self, state):
        [self.sched[i].load_state_dict(s) for i, s in enumerate(state)]
