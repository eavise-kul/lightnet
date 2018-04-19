#
#   Lightnet dataset and dataloading mechanisms
#   Copyright EAVISE
#

import random
import logging
from functools import wraps
import torch
from torch.utils.data.dataset import Dataset as torchDataset
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.dataloader import DataLoader as torchDataLoader
from torch.utils.data.dataloader import default_collate


__all__ = ['Dataset', 'DataLoader', 'list_collate']
log = logging.getLogger(__name__)


class Dataset(torchDataset):
    """ This class is a subclass of the base :class:`torch.utils.data.Dataset`,
    that enables on the fly resizing of the ``input_dim`` with :class:`lightnet.data.DataLoader`.

    Args:
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
    """
    def __init__(self, input_dimension):
        super().__init__()
        self.__input_dim = input_dimension[:2]

    @property
    def input_dim(self):
        """ Dimension that can be used by transforms to set the correct image size, etc.
        This allows transforms to have a single source of truth for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, '_input_dim'):
            return self._input_dim
        return self.__input_dim

    @staticmethod
    def resize_getitem(getitem_fn):
        """ Decorator method that needs to be used around the ``__getitem__`` method. |br|
        This decorator enables the on the fly resizing  of the ``input_dim`` with our :class:`~lightnet.data.DataLoader` class.

        Example:
            .. code:: python
                
                class MyDataSet(ln.data.Dataset):

                    @ln.data.Dataset.resize_getitem
                    def __getitem__(self, index):
                        raise NotImplementedError('Do your thing here')
        """
        @wraps(getitem_fn)
        def wrapper(self, index):
            if not isinstance(index, int):
                self._input_dim = index[0]
                index = index[1]

            ret_val = getitem_fn(self, index)

            if hasattr(self, '_input_dim'):
                del self._input_dim

            return ret_val

        return wrapper


class DataLoader(torchDataLoader):
    """ Lightnet dataloader that enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.

    Note:
        This dataloader only works with :class:`lightnet.data.Dataset` based datasets.
    """
    def __init__(self, *args, resize_range = (10, 19), **kwargs):
        super(DataLoader, self).__init__(*args, **kwargs)
        shuffle = False
        sampler = None
        batch_sampler = None
        if len(args) > 5:
            shuffle = args[2]
            sampler = args[3]
            batch_sampler = args[4]
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if 'batch_sampler' in kwargs:
                batch_sampler = kwargs['batch_sampler']
        elif len(args) > 3:
            shuffle = args[2]
            if 'sampler' in kwargs:
                sampler = kwargs['sampler']
            if 'batch_sampler' in kwargs:
                batch_sampler = kwargs['batch_sampler']
        else:
            if 'shuffle' in kwargs:
                shuffle = kwargs['shuffle']
            if 'sampler' in kwargs:
                sampler = kwargs['sampler']
            if 'batch_sampler' in kwargs:
                batch_sampler = kwargs['batch_sampler']

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = BatchSampler(sampler, self.batch_size, self.drop_last, input_dimension=self.dataset.input_dim)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def change_input_dim(self, multiple=32, random_range=(10, 19)):
        """ This function will compute a new size and update it on the next mini_batch.

        Args:
            multiple (int or tuple, optional): value (or values) to multiply the randomly generated range by; Default **32**
            random_range (tuple, optional): When ``randomize`` is true, this (min, max) tuple sets the range for the randomisation; Default **(10, 19)**

        Note:
            The new size is generated as follows: |br|
            First we compute a random integer inside ``[random_range]``.
            We then multiply that number with the ``multiple`` argument, which gives our final new input size. |br|
            If ``multiple`` is an integer we generate a square size. If you give a tuple of **(width, height)**,
            the size is computed as :math:`rng * multiple[0], rng * multiple[1]`.
        """
        size = random.randint(*random_range)

        if isinstance(multiple, int):
            size = (size * multiple, size * multiple)
        else:
            size = (size * multiple[0], size * multiple[1])
        
        self.batch_sampler.new_input_dim = size


class BatchSampler(torchBatchSampler):
    """ This batch sampler will generate mini-batches of (dim, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`, but it will prepend a dimension,
    whilst ensuring it stays the same across one mini-batch.
    """
    def __init__(self, *args, input_dimension=None, **kwargs):
        super(BatchSampler, self).__init__(*args, **kwargs)
        self.input_dim = input_dimension
        self.new_input_dim = None

    def __iter__(self):
        self.__set_input_dim()
        for batch in super(BatchSampler, self).__iter__():
            yield [(self.input_dim, idx) for idx in batch]
            self.__set_input_dim()

    def __set_input_dim(self):
        """ This function randomly changes the the input dimension of the dataset. """
        if self.new_input_dim is not None:
            log.info(f'Resizing network {self.new_input_dim[:2]}')
            self.input_dim = (self.new_input_dim[0], self.new_input_dim[1])
            self.new_input_dim = None


def list_collate(batch):
    """ Function that collates lists of items together into one list (of lists).
    Use this as the collate function in a Dataloader, if you want to have a list of items as an output, as opposed to tensors (eg. Brambox.boxes).
    """
    items = list(zip(*batch))

    for i in range(len(items)):
        if isinstance(items[i][0], list):
            items[i] = list(items[i])
        else:
            items[i] = default_collate(items[i])

    return items
