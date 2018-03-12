#
#   Lightnet dataset that works with brambox annotations
#   Copyright EAVISE
#   

import os
import copy
import random
from PIL import Image
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.dataloader import DataLoader as torchDataLoader
from torch.utils.data.dataloader import default_collate
import brambox.boxes as bbb

from ..logger import *

__all__ = ['BramboxData', 'DataLoader', 'list_collate']


class BramboxData(Dataset):
    """ Dataset for any brambox parsable annotation format.
        
    Args:
        anno_format (brambox.boxes.format): Annotation format
        anno_filename (list or str): Annotation filename, list of filenames or expandable sequence
        input_dimension (tuple): Tuple containing width,height values
        class_label_map (list): List of class_labels
        identify (function, optional): Lambda/function to get image based of annotation filename or image id; Default **replace/add .png extension to filename/id**
        img_transform (torchvision.transforms.Compose): Transforms to perform on the images
        anno_transform (torchvision.transforms.Compose): Transforms to perform on the annotations
        kwargs (dict): Keyword arguments that are passed to the brambox parser
    """
    def __init__(self, anno_format, anno_filename, input_dimension, class_label_map=None, identify=None, img_transform=None, anno_transform=None, **kwargs):
        super(BramboxData, self).__init__()
        self.__input_dim = input_dimension[:2]
        self.img_tf = img_transform
        self.anno_tf = anno_transform
        if callable(identify):
            self.id = identify
        else:
            self.id = lambda name : os.path.splitext(name)[0] + '.png'

        # Get annotations
        self.annos = bbb.parse(anno_format, anno_filename, identify=lambda f:f, class_label_map=class_label_map, **kwargs)
        self.keys = list(self.annos)

        # Add class_ids
        if class_label_map is None:
            log(Loglvl.WARN, f'No class_label_map given, annotations wont have a class_id values for eg. loss function')
        for k,annos in self.annos.items():
            for a in annos:
                if class_label_map is not None:
                    try:
                        a.class_id = class_label_map.index(a.class_label)
                    except ValueError:
                        log(Loglvl.ERROR, f'{a.class_label} is not found in the class_label_map', ValueError)
                else:
                    a.class_id = 0

        log(Loglvl.VERBOSE, f'Dataset loaded: {len(self.keys)} images')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        """ Get (img, anno) tuple based of index from self.keys """
        if not isinstance(index, int):
            self._input_dim = index[0]
            index = index[1]
        if index >= len(self):
            log(Loglvl.ERROR, f'list index out of range [{index}/{len(self)-1}]', IndexError)

        # Load
        img = Image.open(self.id(self.keys[index]))
        anno = copy.deepcopy(self.annos[self.keys[index]])

        # Transform
        if self.img_tf is not None:
            img = self.img_tf(img)
        if self.anno_tf is not None:
            anno = self.anno_tf(anno)

        if hasattr(self, '_input_dim'):
            del self._input_dim
        return img, anno

    @property
    def input_dim(self):
        """ Dimensions that can be used by transforms to set the correct image size, etc.
        This allows transforms to have a single source of truth for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, '_input_dim'):
            return self._input_dim
        return self.__input_dim


class DataLoader(torchDataLoader):
    """ Lightnet dataloader that enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last, input_dimension=dataset.input_dim)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def change_input_dim(self, value=32, randomize=True):
        """ This function will compute a new size and update it on the next mini_batch.

        Args:
            value (int or tuple, optional): if ``random`` is false this value will be chosen for the new size, else this number represents a multiple for the random size; Default **32**
            randomize (boolean, optional): Whether to randomly compute a new size or set the size given; Default **True**
        """
        if not randomize:
            if isinstance(value, int):
                value = (value, value)
            else:
                value = (value[0], value[1])
            self.batch_sampler.new_input_dim = value
        else:
            if isinstance(value, int):
                size = (random.randint(0,9) + 10) * value 
                size = (size, size)
            else:
                size = ((random.randint(0,9) + 10) * value[0], (random.randint(0,9) + 10) * value[1])
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
            log(Loglvl.VERBOSE, f'Resizing network {self.new_input_dim[:2]}')
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
