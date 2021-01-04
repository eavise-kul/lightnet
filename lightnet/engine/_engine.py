#
#   Base engine class
#   Copyright EAVISE
#

import logging
import signal
from abc import ABC, abstractmethod

import lightnet as ln

__all__ = ['Engine']
log = logging.getLogger(__name__)


class Engine(ABC):
    """ This class removes the boilerplate code needed for writing your training cycle. |br|

    .. warning::
       There are already a lot of PyTorch libraries that are created to ease the creation of a training pipeline.

       In order to limit the burden on the Lightnet dev-team, we will stop working on our own engine
       and instead be slowly transitioning towards `PyTorch Lightning <lightning_>`_. |br|
       This transition will be slow and this engine will thus remain in the codebase for quite some time,
       but no further development will be made to this.

       Besides, PyTorch Lightnig offers a ton of extra functionality
       and is being maintained by a much bigger group of people,
       which allows it to stay up-to-date with recent Deep Learning trends much faster!

    Here is the code that runs when the engine is called:

    .. literalinclude:: /../lightnet/engine/_engine.py
       :language: python
       :pyobject: Engine.__call__
       :dedent: 4

    Args:
        params (lightnet.engine.HyperParameters): Serializable hyperparameters for the engine to work with
        dataloader (torch.utils.data.DataLoader, optional): Dataloader for the training data; Default **None**
        **kwargs (dict, optional): Keywords arguments that will be set as attributes of the engine

    Attributes:
        self.params: HyperParameter object
        self.dataloader: Dataloader object
        self.sigint: Boolean value indicating whether a SIGINT (CTRL+C) was send; Default **False**
        self.*: All values that were passed with the init function and all values from the :class:`~lightnet.engine.HyperParameters` can be accessed in this class

    Note:
        This class expects a `self.dataloader` object to be present. |br|
        You can either pass a dataloader when initializing this class, or you can define it yourself.
        This allows to define `self.dataloader` as a computed property (@property) of your class, opening up a number of different possibilities,
        like eg. computing different dataloaders depending on which epoch you are.

    Note:
        This engine allows to define hook functions to run at certain points in the training *(epoch_start, epoch_end, batch_start, batch_end)*.
        The functions can be defined as class methods of your engine without any extra arguments or as separate functions that take the engine as a single argument.

        There are different functions to register a hook and they can be used as decorator functions or called straight away in code:

        >>> class TrainingEngine(ln.engine.Engine):
        ...     def start(self):
        ...         pass
        ...
        ...     @ln.engine.Engine.epoch_end
        ...     def backup(self):
        ...         pass    # This method will be executed at the end of every epoch
        ...
        ...     @ln.engine.Engine.batch_start(100)
        ...     def update_hyperparams(self):
        ...         pass    # This method will be executed at the start of every 100th batch
        ...
        >>> # Create TrainingEngine object and run it

        >>> def backup(engine):
        ...     pass    # This function will be executed at the end of every Xth batch defined by a backup_rate variable at runtime
        ...
        >>> @ln.engine.Engine.epoch_start
        ... def select_data_subset(engine):
        ...     pass    # This function will be executed at the start of every epoch
        ...
        >>> class TrainingEngine(ln.engine.Engine):
        ...     def start(self):
        ...         if hasattr(self, 'backup_rate') and self.backup_rate is not None:
        ...             self.batch_start(self.backup_rate)(backup)
        ...
        >>> # Create TrainingEngine object and run it
    """
    __init_done = False
    _required_attr = ['network', 'batch_size', 'dataloader']
    _handled_signals = [signal.SIGINT, signal.SIGTERM]
    _epoch_start = {}
    _epoch_end = {}
    _batch_start = {}
    _batch_end = {}

    def __init__(self, params, dataloader=None, **kwargs):
        self.params = params
        if dataloader is not None:
            self.dataloader = dataloader

        # Sigint handling
        self.sigint = False
        for sig in self._handled_signals:
            signal.signal(sig, self.__sigint_handler)

        # Set attributes
        for key in kwargs:
            if not hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                log.warning(f'{key} attribute already exists on engine.')

        self.__init_done = True

    def __call__(self):
        """ Start the training cycle. """
        self.__check_attr()
        self.start()

        log.info('Start training')
        self.network.train()

        idx = 0
        while True:
            # Check if we need to stop training
            if self.quit() or self.sigint:
                log.info('Reached quitting criteria')
                return

            # Epoch Start
            self._run_hooks(self.epoch + 1, self._epoch_start)

            idx %= self.batch_subdivisions
            loader = self.dataloader
            for idx, data in enumerate(loader, idx+1):
                # Batch Start
                if (idx - 1) % self.batch_subdivisions == 0:
                    self._run_hooks(self.batch + 1, self._batch_start)

                # Forward and backward on (mini-)batches
                self.process_batch(data)
                if idx % self.batch_subdivisions != 0:
                    continue

                # Optimizer step
                self.batch += 1     # Should only be called after train, but this is easier to use self.batch in function
                self.train_batch()

                # Batch End
                self._run_hooks(self.batch, self._batch_end)

                # Check if we need to stop training
                if self.quit() or self.sigint:
                    log.info('Reached quitting criteria')
                    return

            # Epoch End
            self.epoch += 1
            self._run_hooks(self.epoch, self._epoch_end)

    def __getattr__(self, name):
        if hasattr(self.params, name):
            return getattr(self.params, name)
        else:
            raise AttributeError(f'{name} attribute does not exist')

    def __setattr__(self, name, value):
        if self.__init_done and name not in dir(self) and hasattr(self.params, name):
            setattr(self.params, name, value)
        else:
            super().__setattr__(name, value)

    def __sigint_handler(self, signal, frame):
        if not self.sigint:
            log.debug('SIGINT/SIGTERM caught. Waiting for gracefull exit')
            self.sigint = True

    def __check_attr(self):
        for attr in self._required_attr:
            if not hasattr(self, attr):
                raise AttributeError(f'Engine requires attribute [{attr}] (as an engine or hyperparameter attribute)')

        if not hasattr(self, 'mini_batch_size'):
            log.warning('No [mini_batch_size] attribute found, setting it to [batch_size]')
            self.mini_batch_size = self.batch_size
        elif self.batch_size % self.mini_batch_size != 0 or self.mini_batch_size > self.batch_size:
            raise ValueError('batch_size should be a multiple of mini_batch_size')

    def log(self, msg):
        """ Log messages about training and testing.
        This function will automatically prepend the messages with **TRAIN** or **TEST**.

        Args:
            msg (str): message to be printed
        """
        if self.network.training:
            log.train(msg)
        else:
            log.test(msg)

    def _run_hooks(self, value, hooks):
        """ Internal method that will execute registered hooks. """
        keys = list(hooks.keys())
        for k in keys:
            if value % k == 0:
                for fn in hooks[k]:
                    if hasattr(fn, '__self__'):
                        fn()
                    else:
                        fn(self)

    @classmethod
    def epoch_start(cls, interval=1):
        """ Register a hook to run at the start of an epoch.

        Args:
            interval (int, optional): Number dictating how often to run the hook; Default **1**

        Note:
            The `self.epoch` attribute contains the number of processed epochs,
            and will thus be one lower than the epoch you are currently starting.
            For example, when starting training with the very first epoch,
            the `self.epoch` attribute will be set to **0** during any `epoch_start` hook. |br|
            However, the `interval` argument will be computed with the correct epoch number (ic. `self.epoch` + 1).
        """
        def decorator(fn):
            if interval in cls._epoch_start:
                cls._epoch_start[interval].append(fn)
            else:
                cls._epoch_start[interval] = [fn]
            return fn

        return decorator

    @classmethod
    def epoch_end(cls, interval=1):
        """ Register a hook to run at the end of an epoch.

        Args:
            interval (int, optional): Number dictating how often to run the hook; Default **1**
        """
        def decorator(fn):
            if interval in cls._epoch_end:
                cls._epoch_end[interval].append(fn)
            else:
                cls._epoch_end[interval] = [fn]
            return fn

        return decorator

    @classmethod
    def batch_start(cls, interval=1):
        """ Register a hook to run at the start of a batch.

        Args:
            interval (int, optional): Number dictating how often to run the hook; Default **1**

        Note:
            The `self.batch` attribute contains the number of processed batches,
            and will thus be one lower than the batch you are currently starting.
            For example, when starting training with the very first batch,
            the `self.batch` attribute will be set to **0** during any `batch_start` hook. |br|
            However, the `interval` argument will be computed with the correct batch number (ic. `self.batch` + 1).
        """
        def decorator(fn):
            if interval in cls._batch_start:
                cls._batch_start[interval].append(fn)
            else:
                cls._batch_start[interval] = [fn]
            return fn

        return decorator

    @classmethod
    def batch_end(cls, interval=1):
        """ Register a hook to run at the end of a batch.

        Args:
            interval (int, optional): Number dictating how often to run the hook; Default **1**
        """
        def decorator(fn):
            if interval in cls._batch_end:
                cls._batch_end[interval].append(fn)
            else:
                cls._batch_end[interval] = [fn]
            return fn

        return decorator

    @property
    def batch_subdivisions(self):
        """ Get number of mini-batches per batch.

        Return:
            int: Computed as self.batch_size // self.mini_batch_size
        """
        return self.batch_size // self.mini_batch_size

    def start(self):
        """ First function that gets called when starting the engine. |br|
        Any required setup code can come in here.
        """
        pass

    @abstractmethod
    def process_batch(self, data):
        """ This function should contain the code to process the forward and backward pass of one (mini-)batch.

        Args:
            data: The data that comes from your dataloader

        Note:
            If you are working with mini-batches, you should pay attention to how you process your loss and backwards function. |br|
            PyTorch accumulates gradients when performing multiple backward() calls before using your optimizer.
            However, usually your loss function performs some kind of average over your batch-size (eg. reduction='mean' in a lot of default pytorch functions).
            When that is the case, you should also average your losses over the mini-batches, by dividing your resulting loss:

            .. code:: bash

                loss = loss_function(output, target) / self.batch_subdivisions
                loss.backward()
        """
        pass

    @abstractmethod
    def train_batch(self):
        """ This function should contain the code to update the weights of the network. |br|
        Statistical computations, performing backups at regular intervals, etc. also happen here.
        """
        pass

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
        return False
