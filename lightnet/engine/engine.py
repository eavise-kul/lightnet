#
#   Base engine class
#   Copyright EAVISE
#

from statistics import mean
import signal
import sys
import torch
try:
    import visdom
except ImportError:
    visdom = None


import lightnet as ln
from ..logger import *

__all__ = ['Engine']


class Engine:
    """ This class removes the boilerplate code needed for writing your training cycle. |br|
    Here is the code that runs when the engine is called:
    
    .. literalinclude:: /../lightnet/engine/engine.py
       :language: python
       :pyobject: Engine.__call__
       :dedent: 4

    Args:
        network (lightnet.network.Darknet): Lightnet network to train
        optimizer (torch.optim): Optimizer for the network
        visdom (dict, optional): Set this dict with options for starting up visdom. If set to None, visualisation with visdom is disabled; Default **None**

    Attributes:
        self.network: Lightnet network
        self.optimizer: Torch optimizer
        self.batch_size: Number indicating batch_size; Default **1**
        self.batch_subdivisions: How to subdivide batch; Default **1**
        self.max_batch: Maximum number of batches to process; Default **None**
        self.test_rate: How often to run test; Default **None**
        self.visdom: Visdom object used to plot data; Default **None**
        self.sigint: Boolean value indicating whether a SIGINT (CTRL+C) was send; Default **False**
    """
    batch_size = 1
    batch_subdivisions = 1
    max_batch = None
    test_rate = None

    def __init__(self, network, optimizer, visdom_opts=None):
        self.network = network
        self.optimizer = optimizer

        self.__lr = self.optimizer.param_groups[0]['lr']
        self.__rates = {}

        self.sigint = False
        signal.signal(signal.SIGINT, self.__sigint_handler)

        self.__log = ln.logger.Logger()
        self.__log.color = False
        self.__log.level = 0
        self.__log.lvl_msg = ['[TRAIN]   ', '[TEST]    ']

        if visdom_opts is not None:
            self.visdom = visdom.Visdom(visdom_opts)
        else:
            self.visdom = None
    
    def __call__(self):
        """ Start the training cycle. """
        self.start()
        if self.test_rate is not None:
            last_test = self.batch - (self.batch % self.test_rate)

        while True:
            log(Loglvl.DEBUG, 'Starting train epoch')
            self.network.train()
            self.train()

            self.update_rates()

            if self.quit() or self.sigint:
                log(Loglvl.VERBOSE, 'Reached quitting criteria')
                break

            if self.test_rate is not None and self.batch - last_test >= self.test_rate:
                log(Loglvl.DEBUG, 'Starting test epoch')
                last_test += self.test_rate
                self.network.eval()
                self.test()

    @property
    def batch(self):
        """ Get current batch number.

        Return:
            int: Computed as self.network.seen // self.batch_size
        """
        return self.network.seen // self.batch_size

    @property
    def mini_batch_size(self):
        """ Get the size of one mini_batch

        Return:
            int: Computed as self.batch_size // self.batch_subdivisions
        """
        return self.batch_size // self.batch_subdivisions

    @property
    def learning_rate(self):
        """ Get and set the learning rate
            
        Args:
            lr (Number): Set the learning rate for all values of optimizer.param_groups[i]['lr']

        Return:
            Number: The current learning rate
        """
        return self.__lr

    @learning_rate.setter
    def learning_rate(self, lr):
        self.__lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def log(self, msg):
        """ Log messages about training and testing.
        This function will automatically prepend the messages with **[TRAIN]** or **[TEST]**.

        Args:
            msg (str): message to be printed
        """
        if self.network.training:
            self.__log(0, msg)
        else:
            self.__log(1, msg)

    def add_rate(self, name, steps, values, default=None):
        """ Add a rate to the engine. Rates are values that change according to the current batch number.

        Args:
            name (str): Name that will be used for the attribute. You can access the value with self.name
            steps (list): Batches at which the rate should change
            values (list): New values that will be used for the attribute
            default (optional): Default value to use for the rate; Default **None**

        Example:
            >>> import lightnet as ln
            >>> eng = ln.engine.Engine(...)
            >>> eng.add_rate('test_rate', [1000, 5000], [100, 500], 50)
            >>> eng.add_rate('learning_rate', [100, 1000, 10000], [.01, .001, .0001]) # Learning rate already has a value
        """
        if default is not None or not hasattr(self, name):
            setattr(self, name, default)
        if name in self.__rates:
            log(Loglvl.WARN, f'{name} rate was already used, overwriting...')

        if len(steps) > len(values):
            diff = len(steps) - len(values)
            values = values + diff * [values[-1]]
            log(Loglvl.WARN, f'{name} has more steps than values, extending values to {values}')
        elif len(steps) < len(values):
            values = values[:len(steps)]
            log(Loglvl.WARN, f'{name} has more values than steps, shortening values to {values}')

        self.__rates[name] = (steps, values)

    def update_rates(self):
        """ Update rates according to batch size. |br|
        This function gets automatically called every epoch,
        but to be entirely correct you should also call this function every batch in your training cycle.
        """
        for key, (steps,values) in self.__rates.items():
            new_rate = None
            for i in range(len(steps)):
                if self.batch >= steps[i]:
                    new_rate = values[i]
                else:
                    break

            if new_rate is not None and new_rate != getattr(self, key):
                log(Loglvl.VERBOSE, f'Adjusting {key} [{new_rate}]')
                setattr(self, key, new_rate)

    def start(self):
        """ First function that gets called when starting the engine. |br|
            Use it to create your dataloader, set the correct starting values for your rates, etc.
        """
        self.update_rates()

    def train(self):
        """ Code to train one epoch should come in this function. """
        raise NotImplementedError

    def test(self):
        """ This function should contain the code to perform one evaluation on your test-set. """
        raise NotImplementedError

    def quit(self):
        """ This function gets called after every training epoch and decides if the training cycle continues.

        Return:
            Boolean: Whether are not to stop the training cycle

        Note:
            This function gets called before checking the ``self.sigint`` attribute.
            This means you can also check this attribute in this function. |br|
            If it evaluates to **True**, you know the program will exit after this function and you can thus
            perform the necessary actions (eg. save final weights).
        """
        if self.max_batch is not None:
            return self.batch >= self.max_batch
        else:
            return False

    def __sigint_handler(self, signal, frame):
        if not self.sigint:
            log(Loglvl.DEBUG, 'SIGINT caught. Waiting for gracefull exit')
            self.sigint = True
