#
#   Learning Rate Scheduling
#   Copyright EAVISE
#

import torch
import logging

__all__ = ['SchedulerCompositor']
log = logging.getLogger(__name__)


class SchedulerCompositor:
    """ This class can be used to schedule schedulers to run at different moments on the same parameters of a network. |br|
    This compositor has a notion of count values. These can be batch number, epoch number, etc. and dictate when each scheduler is being used.

    Args:
        *args (list of tuples): This class gets initialized with tuples of ``(count, scheduler)``, which determine when to start using which scheduler

    Example:
        >>> class DummyScheduler:
        ...     " Dummy scheduler that does nothing but print it's id value. "
        ...     def __init__(self, id_value):
        ...         self.id = id_value;
        ...     def step(self, count_value=0):
        ...         print(f'{count_value} - Dummy Scheduler: {self.id}')
        >>> s = ln.engine.SchedulerCompositor(
        ...     (0, DummyScheduler('start')),
        ...     (2, DummyScheduler('middle')),
        ...     (3, DummyScheduler('end')),
        ... )
        >>> for i in range(5):
        ...     s.step(i, count_value=i)
        0 - Dummy Scheduler: start
        1 - Dummy Scheduler: start
        2 - Dummy Scheduler: middle
        3 - Dummy Scheduler: end
        4 - Dummy Scheduler: end
    """
    def __init__(self, *args):
        if len(args) == 0:
            raise ValueError('Compositor requires at least one scheduler')

        self.counts, self.sched = zip(*args)
        if not all(c1 < c2 for c1, c2 in zip(self.counts, self.counts[1:])):
            raise ValueError('Count values need to be strictly increasing')

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ['
        clen = max(len(str(c)) for c in self.counts)
        for i in range(len(self.counts)):
            if hasattr(self.sched[i], '__name__'):
                name = self.sched[i].__name__
            else:
                name = self.sched[i].__class__.__name__

            format_string += f'\n  {self.counts[i]:>{clen}}:  {name}'
        format_string += '\n]'
        return format_string

    def step(self, count, **kwargs):
        """ Stepping function that will select a scheduler and run it.

        Args:
            count (int): Count value that will determine which scheduler to run
            **kwargs (dict, optional): Extra arguments that will be passed on to the step function of the scheduler itself.
        """
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
        return state_dict

    def load_state_dict(self, state, strict=True):
        [self.sched[i].load_state_dict(s) for i, s in enumerate(state)]

    def to(self, device):
        """ Cast schedulers to a certain device.

        Args:
            device (torch.device): Device to cast the scheduler to.
        """
        for sched in self.sched:
            for param in sched.__dict__.values():
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(device)
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(device)
