#
#   HyperParemeters container class
#   Copyright EAVISE
#

import logging
import importlib.util
from collections import Iterable
import torch

__all__ = ['HyperParameters']
log = logging.getLogger(__name__)


class HyperParameters:
    """ This class is a container for training hyperparameters.
    It allows to save the state of a training and reload it at a later stage.

    Args:
        network (torch.nn.Module, optional): Network module; Default **None**
        loss (torch.nn.Module, optional): Loss module; Default **None**
        optimizers (torch.optim.Optimizer or list of torch.optim.Optimizer, optional): Optimizer(s) for the network; Default **None**
        schedulers (torch.optim._LRScheduler or list of torch.optim._LRScheduler, optional): Scheduler(s) for the network; Default; **None**
        batch_size (int, optional): Size of a batch for training; Default **1**
        mini_batch_size (int, optional): Size of a mini-batch for training; Default **batch_size**
        **kwargs (dict, optional): Keywords arguments that will be set as attributes of the instance and serialized as well

    Attributes:
        self.batch: Number of batches processed; Gets initialized to **0**
        self.epoch: Number of epochs processed; Gets initialized to **0**
        self.*: All arguments passed to the initialization function can be accessed as attributes of this object

    Note:
        If you pass a ``kwarg`` that starts with an **_**,
        the parameter class will store it as a regular property without the leading **_**, but it will not serialize this variable.
        This allows you to store all parameters in this object, regardless of whether you want to serialize it.

        This also works when assigning new values after the object creation:
            >>> param = ln.engine.HyperParameters()
            >>> param._dummy = 666  # This wil store it as .dummy and add it to the list that does not need to be serialized.
            >>> print(param.dummy)  # 666

    Note:
        ``batch_size`` must be a multiple of ``mini_batch_size``.

    Warning:
        The :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` LR scheduler does not follow the regular LR scheduler classes.
        Therefore this class will not work with this HyperParameter function.
        If you are using this class, save it as a regular variable, through another name (via kwargs). |br|
        I know this solution is suboptimal, but I am not willing to hack around this, as I believe this is something dirty in the torch codebase itself and should be handled there.
    """
    __init_done = False

    def __init__(self, network=None, optimizers=None, schedulers=None, batch_size=1, mini_batch_size=None, **kwargs):
        self.network = network
        self.batch_size = batch_size
        self.batch = 0
        self.epoch = 0

        if mini_batch_size is None:
            self.mini_batch_size = batch_size
        elif batch_size % mini_batch_size != 0 or mini_batch_size > batch_size:
            raise ValueError('batch_size should be a multiple of mini_batch_size')
        else:
            self.mini_batch_size = mini_batch_size

        if optimizers is None or isinstance(optimizers, Iterable):
            self.optimizers = optimizers
        else:
            self.optimizers = [optimizers]

        if schedulers is None or isinstance(schedulers, Iterable):
            self.schedulers = schedulers
        else:
            self.schedulers = [schedulers]

        self.__no_serialize = ['network', 'loss', 'optimizers', 'schedulers']
        for key in kwargs:
            if key.startswith('_'):
                serialize = False
                val = kwargs[key]
                key = key[1:]
            else:
                serialize = True
                val = kwargs[key]

            if not hasattr(self, key):
                setattr(self, key, val)
                if not serialize:
                    self.__no_serialize.append(key)
            else:
                log.error(f'{key} attribute already exists as a HyperParameter and will not be overwritten.')

        self.__init_done = True

    def __setattr__(self, item, value):
        """ Store extra variables in this container class.
        This custom function allows to store objects after creation and mark whether are not you want to serialize them,
        by prefixing them with an underscore.
        """
        if item in self.__dict__ or not self.__init_done:
            super().__setattr__(item, value)
        elif item[0] == '_':
            if item[1:] in self.__dict__:
                log.error(f'{item} already stored in this object, not overwriting! Use {item[1:]} to access and modify it.')
                return
            self.__no_serialize.append(item[1:])
            super().__setattr__(item[1:], value)
        else:
            super().__setattr__(item, value)

    def __repr__(self):
        """ Print all values stored in the object.
        Objects that will not be serialized are marked with an asterisk.
        """
        s = f'{self.__class__.__name__}(\n'
        if self.network is None:
            s += '  [No Network]\n  '
        else:
            s += f'  {self.network.__class__.__name__}\n  '

        if self.optimizers is None:
            s += '[No Optimizers]\n  '
        else:
            for i, o in enumerate(self.optimizers):
                if i != 0:
                    s += ', '
                s += o.__class__.__name__
            s += '\n  '

        if self.schedulers is None:
            s += '[No Schedulers]\n'
        else:
            for i, sc in enumerate(self.schedulers):
                if i != 0:
                    s += ', '
                s += sc.__class__.__name__
            s += '\n'

        for k in sorted(self.__dict__.keys()):
            if k.startswith('_HyperParameters__') or k in ['network', 'optimizers', 'schedulers']:
                continue

            val = self.__dict__[k]
            valrepr = str(val)
            if '\n' in valrepr:
                valrepr = val.__class__.__name__
            if k in self.__no_serialize:
                k += '*'

            s += f'\n  {k} = {valrepr}'

        return s + '\n)'

    @classmethod
    def from_file(cls, path, variable='params', **kwargs):
        """ Create a HyperParameter object from a dictionary in an external configuration file.
        This function will import a file by its path and extract a variable to use as HyperParameters.

        Args:
            path (str or path-like object): Path to the configuration python file
            variable (str, optional): Variable to extract from the configuration file; Default **'params'**
            **kwargs (dict, optional): Extra parameters that are passed to the extracted variable if it is a callable object

        Note:
            The extracted variable can be one of the following:

            - :class:`lightnet.engine.HyperParameters`: This object will simply be returned
            - ``dictionary``: The dictionary will be expanded as the parameters for initializing a new :class:`~lightnet.engine.HyperParameters` object
            - ``callable``: The object will be called with the optional kwargs and should return either a :class:`~lightnet.engine.HyperParameters` object or a ``dictionary``
        """
        try:
            spec = importlib.util.spec_from_file_location('lightnet.cfg', path)
            cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg)
        except AttributeError as err:
            raise ImportError(f'Failed to import the file [{path}]. Are you sure it is a valid python file?') from err

        try:
            params = getattr(cfg, variable)
        except AttributeError as err:
            raise AttributeError(f'Configuration variable [{variable}] not found in file [{path}]') from err

        if callable(params):
            params = params(**kwargs)

        if isinstance(params, cls):
            return params
        elif isinstance(params, dict):
            return cls(**params)
        else:
            raise TypeError(f'Unkown type for configuration variable {variable} [{type(params).__name__}]. This variable should be a dictionary or lightnet.engine.HyperParameters object.')

    @property
    def optimizer(self):
        """ Convenience property to access the first optimizer.

        Return:
            torch.optim.Optimizer: self.optimizers[0]
        """
        return self.optimizers[0]

    @property
    def scheduler(self):
        """ Convenience property to access the first scheduler.

        Return:
            torch.optim.lr_scheduler: self.schedulers[0]
        """
        return self.schedulers[0]

    @property
    def batch_subdivisions(self):
        """ Get number of mini-batches per batch.

        Return:
            int: Computed as self.batch_size // self.mini_batch_size
        """
        return self.batch_size // self.mini_batch_size

    def add_optimizer(self, optimizer):
        if self.optimizers is None:
            self.optimizers = [optimizer]
        else:
            self.optimizers.append(optimizer)

    def add_scheduler(self, scheduler):
        if self.schedulers is None:
            self.schedulers = [scheduler]
        else:
            self.schedulers.append(scheduler)

    def save(self, filename, store_optim_sched=True):
        """ Serialize all the hyperparameters to a pickle file. |br|
        The network, optimizers and schedulers objects are serialized using their ``state_dict()`` functions.

        Args:
            filename (str or path): File to store the hyperparameters
            store_optim_sched (bool, optional): Whether to store the optimizers and schedulers; Default **True**
        """
        state = {k: v for k, v in vars(self).items() if k not in self.__no_serialize}

        if self.network is not None:
            state['network'] = self.network.state_dict()
        if store_optim_sched:
            if self.optimizers is not None:
                state['optimizers'] = [optim.state_dict() for optim in self.optimizers]
            if self.schedulers is not None:
                state['schedulers'] = [sched.state_dict() for sched in self.schedulers]

        torch.save(state, filename)

    def load(self, filename, strict=False):
        """ Load the hyperparameters from a serialized pickle file.

        Warning:
            This function expects a serialized file with hyperparameters for the same network, optimizers and schedulers.
        """
        state = torch.load(filename, lambda storage, loc: storage)

        if self.network is not None and 'network' in state:
            self.network.load_state_dict(state.pop('network'), strict=strict)
        if self.optimizers is not None and 'optimizers' in state:
            optim_state = state.pop('optimizers')
            for i, optim in enumerate(self.optimizers):
                optim.load_state_dict(optim_state[i])
        if self.schedulers is not None and 'schedulers' in state:
            sched_state = state.pop('schedulers')
            for i, sched in enumerate(self.schedulers):
                sched.load_state_dict(sched_state[i])

        for key, value in state.items():
            setattr(self, key, value)

    def to(self, device):
        """ Cast the parameters from the network, optimizers and schedulers to a given device. """
        for key, value in self.__dict__.items():
            if key not in ('network', 'optimizers', 'schedulers') and isinstance(value, torch.nn.Module):
                value.to(device)

        if self.network is not None:
            self.network.to(device)

        if self.optimizers is not None:
            for optim in self.optimizers:
                for param in optim.state.values():
                    if isinstance(param, torch.Tensor):
                        param.data = param.data.to(device)
                        if param._grad is not None:
                            param._grad.data = param._grad.data.to(device)
                    elif isinstance(param, dict):
                        for subparam in param.values():
                            if isinstance(subparam, torch.Tensor):
                                subparam.data = subparam.data.to(device)
                                if subparam._grad is not None:
                                    subparam._grad.data = subparam._grad.data.to(device)

        if self.schedulers is not None:
            for sched in self.schedulers:
                for param in sched.__dict__.values():
                    if isinstance(param, torch.Tensor):
                        param.data = param.data.to(device)
                        if param._grad is not None:
                            param._grad.data = param._grad.data.to(device)
