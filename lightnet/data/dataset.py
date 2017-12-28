#
#   Lightnet dataset that works with brambox annotations
#   Copyright EAVISE
#   

import os
import copy
import random
from PIL import Image
import torch.multiprocessing as multiprocessing
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
import brambox.boxes as bbb

from ..logger import *

__all__ = ['BramboxData', 'list_collate']


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
        self.input_dim = multiprocessing.Array('i', input_dimension[:2])
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
            log(Loglvl.WARN, f'No class_label_map given, annotations wont have a class_id for the loss function')
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

        return img, anno

    def change_input_dim(self, multiple=32):
        """ This function randomly changes the the input dimension of the dataset (can be used by letterbox).
        It changes the **self.input_dim** variable to be a random number between **(10-19)*multiple**.

        Args:
            multiple (int, optional): Factor to change the random new size; Default **32**
        """
        size = (random.randint(0,9) + 10) * multiple 
        log(Loglvl.VERBOSE, f'Resizing network [{size}]')

        self.input_dim[0] = size
        self.input_dim[1] = size


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
