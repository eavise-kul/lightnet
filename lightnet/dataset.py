#
#   Darknet specific dataset
#   Copyright EAVISE
#   

import os
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as tf
from PIL import Image
import brambox.boxes as bbb

from .transforms import *
from .logger import *

__all__ = ['BramboxData', 'DarknetData', 'bbb_collate']


class BramboxData(Dataset):
    """ Dataset for any brambox parsable annotation format
        
        anno_format         Annotation format from brambox.boxes.format
        anno_filename       Annotation filename, list of filenames or expandable sequence
        identify            Lambda/function to get image based of annotation filename or image id
                            DEFAULT: replace/add .png extension to filename/id
        img_transform       Transforms to perform on the images
        anno_transform      Transforms to perform on the annotations
        train               Boolean value indicating whether to return the annotation or not
        kwargs              Keyword arguments that are passed to the brambox parser
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
        anno = self.annos[self.keys[index]]

        # Transform
        if self.img_tf is not None:
            img = self.img_tf(img)
        if self.anno_tf is not None:
            anno = self.anno_tf(anno)

        return img, anno


class DarknetData(BramboxData):
    """ Dataset that works with darknet files and performs the same data augmentations
        
        data_file       File containing path to image files (relative from where command is run)
        network         Network that will be used with dataset (needed for network input dimension)
        train           Boolean indicating whether to return the annotation
        augment         Boolean indicating whether you want data augmentation
        jitter          Determines random crop sizes
        flip            Determines whether image will be flipped
        hue             Determines hue shift
        saturation      Determines saturation shift
        value           Determines value (exposure) shift
    """
    def __init__(self, data_file, network, train=True, augment=True, jitter=.2, flip=.5, hue=.1, saturation=1.5, value=1.5, class_label_map=None):
        with open(data_file, 'r') as f:
            self.img_paths = f.read().splitlines()

        # Prepare variables for brambox init
        anno_format = 'anno_darknet'
        self.anno_paths = [os.path.splitext(p)[0]+'.txt' for p in self.img_paths]
        identify = lambda name : self.img_paths[self.anno_paths.index(name)]
        
        lb  = Letterbox(network)
        rf  = RandomFlip(flip)
        rc  = RandomCrop(jitter, True)
        hsv = HSVShift(hue, saturation, value)
        at  = AnnoToTensor(network)
        it  = tf.ToTensor()
        if augment:
            img_tf = tf.Compose([hsv, rc, rf, lb, it])
            anno_tf = tf.Compose([rc, rf, lb])
        else:
            img_tf = tf.Compose([lb, it])
            anno_tf = tf.Compose([lb])
        if train:
            anno_tf.transforms.append(at)
        
        first_img = Image.open(self.img_paths[0])
        w, h = first_img.size
        kwargs = { 'image_width': w, 'image_height':h, 'class_label_map': class_label_map }

        super(DarknetData, self).__init__(anno_format, self.anno_paths, identify, img_tf, anno_tf, **kwargs)

        # Memory optimisation: set AnnoToTensor maximum
        self.max_anno = max([len(anno) for _,anno in self.annos.items()])
        if train:
            at.max = self.max_anno


def bbb_collate(batch):
    if isinstance(batch[0][1], list):
        img,anno = zip(*batch)
        img = default_collate(img)
        return img, list(anno)

    return default_collate(batch)
