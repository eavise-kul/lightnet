#
#   Base engine class
#   Copyright EAVISE
#

from statistics import mean
import signal
import sys
import torch

import lightnet as ln
from ..logger import *

__all__ = ['Engine']


class Engine:
    """ This class removes the boilerplate code needed for writing the training code.
    Here is the code that runs when the engine is called
    
    .. literalinclude:: /../lightnet/engine/engine.py
       :language: python
       :pyobject: Engine.__call__
       :dedent: 4

    Args:
        network (lightnet.network.Darknet): Lightnet network to train
        optimizer (torch.optim): Optimizer for the network
        trainset (torch.utils.data.Dataset): Dataset with training images and annotations
        testset (torch.utils.data.Dataset, optional): Dataset with images and annotation to test; Default **None**
        cuda (Boolean, optional): Whether to use cuda; Default **False**
        visdom (dict, optional): Set this dict with options for starting up visdom. If set to None, visualisation with visdom is disabled; Default **None**
        **kwargs (dict, optional): Extra arguments that are set as attributes to the engine

    Attributes:
        self.network: Lightnet network
        self.optimizer: Torch optimizer
        self.trainset: Torch dataset for training
        self.testset: Torch dataset for testing; Default **None**
        self.cuda: Boolean indicating whether to use cuda
        self.batch_size: Number indicating batch_size
        self.batch_subdivisions: How to subdivide batch
        self.max_batch: Maximum number of batches to process
        self.test_rate: How often to run test
        self.visdom: Visdom object used to plot data
        self.sigint: Boolean value indicating whether a SIGINT (CTRL+C) was send.

        self.batch (computed): Current batch number
        self.mini_batch_size (computed): Size of one mini-batch, according to batch_size and batch_subdivisions
        self.learning_rate (computed): Property to set and get learning rate of the optimizer

    Note:
        Some preset values of the engine can be overwritten with kwargs:

        - batch_size: size of one batch
        - batch_subdivisions: number of subdivisions needed for one batch
        - max_batch: maximum number of batches to train for
        - test_rate: how often to run testset
    """

    __allowed = ('batch_size', 'batch_subdivisions', 'max_batch', 'test_rate')

    def __init__(self, network, optimizer, trainset, testset=None, cuda=False, visdom=None, **kwargs):
        self.network = network
        self.optimizer = optimizer
        self.trainset = trainset
        self.testset = testset
        self.cuda = cuda

        self.__lr = self.optimizer.param_groups[0]['lr']
        self.__rates = {}

        self.sigint = False
        signal.signal(signal.SIGINT, self.__sigint_handler)

        self.__log = ln.logger.Logger()
        self.__log.color = False
        self.__log.level = 0
        self.__log.lvl_msg = ['[TRAIN]   ', '[TEST]    ']

        if visdom is not None:
            from .visual import Visualisation
            self.__vis = Visualisation(visdom)
            self.visdom = self.__vis.vis
        else:
            self.__vis = None
            self.visdom = None
    
        self.batch_size = 64
        self.batch_subdivisions = 8
        self.max_batch = None
        self.test_rate = 50

        for key,val in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, val)
            elif key in self.__class__.__allowed:
                log(Loglvl.DEBUG, f'Attribute [{key}] already exists, overwriting with kwarg value [{val}]')
                setattr(self, key, val)
            else:
                log(Loglvl.WARN, f'Attribute [{key}] already exists, keeping old value [{getattr(self, key)}]')
    
    def __call__(self):
        """ Start the training cycle. """
        self.start()

        last_test = self.batch - (self.batch % self.test_rate)
        while True:
            log(Loglvl.DEBUG, 'Starting train epoch')
            self.network.train()
            self.train()

            self.update_rates()

            if self.quit() or self.sigint:
                log(Loglvl.VERBOSE, 'Reached quitting criteria')
                break

            if self.testset is not None and self.batch - last_test >= self.test_rate:
                log(Loglvl.DEBUG, 'Starting test epoch')
                last_test += self.test_rate
                self.network.eval()
                self.test()

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

    def visual(self, **kwargs):
        """ visualisation wrapper function.
        This function will call the visdom functions on the right windows and titles.

        Args:
            kwargs (dict): Call this functions with key-value pairs representing the arguments of the functions of :class:`~lightnet.engine.Visualisation`

        Note:
            The keyword arguments should have all parameters that the underlining functions from :class:`~lightnet.engine.Visualisation` requires.
            This function will automatically infer the correct function, from the arguments passed.
            Some parameters (eg. batch number) will be automatically computed from the engine data. |br|
            Besides the parameters of the function, you can also pass an **opts** keyword argument.
            This keyword needs to be a dictionary, containing extra options that are passed along to visdom_.
        """
        if self.__vis is not None:
            if self.network.training:
                win = 'Train'
            else:
                win = 'Test'

            if 'opts' in kwargs:
                options = kwargs['opts']
            else:
                options = {}

            if 'pr' in kwargs:
                if not 'title' in options:
                    self.__vis.pr(kwargs['pr'], f'{win}_pr', title=f'PR-curve [{self.batch}]', **options)
                else:
                    self.__vis.pr(kwargs['pr'], f'{win}_pr', **options)
            elif 'loss' in kwargs:
                self.__vis.loss(kwargs['loss'], self.batch, f'{win}_loss', kwargs['name'], title=f'{win} loss', **options)
            else:
                log(Loglvl.WARN, 'Could not find out what visualisation function to use.')

    @property
    def batch(self):
        """ Get current batch number.

        Return:
            int: Computed as self.network.seen // self.batch_size
        """
        return self.network.seen // self.batch_size

    @property
    def mini_batch_size(self):
        """ Get size of one mini-batch.
        
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
            >>> eng.add_rate('learning_rate', [100, 500, 35000], [.001, .0001, .00001], .0001)
            >>> eng.add_rate('test_rate', [1000, 5000], [100, 500]) # test_rate already has default value of 50
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
        """ Update rates according to batch size.
        This function gets automatically called every epoch, but to be entirely correct,
        you should also call this function every batch in your training cycle.
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

    def train(self):
        """ Training loop code.

        Args:
            idx (int): Mini-batch number
            data: Return Value from the trainset; usually a tuple (data, target)
        """
        raise NotImplementedError

    def test(self):
        """ Test loop code.

        Args:
            idx (int): Mini-batch number
            data: Return value from the testset; usually a tuple (data, target)
        """
        raise NotImplementedError

    def start(self):
        """ First function that gets called when starting the engine.
            Use it to set correct starting values for learning rate, test rate, etc.
        """
        pass

    def quit(self):
        """ This function gets called after every training epoch and decides if the training cycle continues.

        Return:
            Boolean: Whether are not to stop the training cycle
        """
        if self.max_batch is not None:
            return self.batch >= self.max_batch
        else:
            return False

    def __sigint_handler(self, signal, frame):
        if not self.sigint:
            log(Loglvl.DEBUG, 'SIGINT caught. Waiting for gracefull exit')
            self.sigint = True
