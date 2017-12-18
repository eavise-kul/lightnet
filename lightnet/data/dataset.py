#
#   Lightnet dataset that works with brambox annotations
#   Copyright EAVISE
#   

import os
import copy
from PIL import Image
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
        identify (function, optional): Lambda/function to get image based of annotation filename or image id; Default **replace/add .png extension to filename/id**
        img_transform (torchvision.transforms.Compose): Transforms to perform on the images
        anno_transform (torchvision.transforms.Compose): Transforms to perform on the annotations
        kwargs (dict): Keyword arguments that are passed to the brambox parser
    """
    def __init__(self, anno_format, anno_filename, identify=None, img_transform=None, anno_transform=None, **kwargs):
        super(BramboxData, self).__init__()
        self.img_tf = img_transform
        self.anno_tf = anno_transform
        if identify is not None and callable(identify):
            self.id = identify
        else:
            self.id = lambda name : os.path.splitext(name)[0] + '.png'

        # Get annotations
        self.annos = bbb.parse(anno_format, anno_filename, identify=lambda f:f, **kwargs)
        self.keys = list(self.annos)
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
