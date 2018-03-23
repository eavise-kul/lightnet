#
#   Darknet dataset
#   Copyright EAVISE
#

import os
from PIL import Image
from torchvision import transforms as tf

import lightnet.data as lnd

__all__ = ['DarknetData']


class DarknetData(lnd.BramboxData):
    """ Dataset that works with darknet files and performs the same data augmentations.
    You must use this dataset with the :meth:`~lightnet.data.list_collate` function in a dataloader.
    If you enable the data augmentation you must also use the :class:`~lightnet.data.DataLoader` class as dataloader.
        
    Args:
        data_file (str): File containing path to image files (relative from where command is run)
        augment (Boolean, optional): Whether or not you want data augmentation; Default **True**
        input_dimension (tuple): Input dimension of the network width,height; Default **416,416**
        jitter (Number [0-1], optional): Determines random crop sizes; Default **0.2**
        flip (Number [0-1], optional): Determines whether image will be flipped; Default **0.5**
        hue (Number, optional): Determines hue shift; Default **0.1**
        saturation (Number, optional): Determines saturation shift; Default **1.5**
        value (Number, optional): Determines value (exposure) shift; Default **1.5**
        class_label_map (list, optional): class label map to convert class names to an index; Default **None**

    Return:
        (tuple): image_tensor, list of brambox boxes
    """
    def __init__(self, data_file, augment=True, input_dimension=(416,416), jitter=.2, flip=.5, hue=.1, saturation=1.5, value=1.5, class_label_map=None):
        with open(data_file, 'r') as f:
            self.img_paths = f.read().splitlines()

        # Prepare variables for brambox init
        anno_format = 'anno_darknet'
        self.anno_paths = [os.path.splitext(p)[0]+'.txt' for p in self.img_paths]
        identify = lambda name : self.img_paths[self.anno_paths.index(name)]
        
        lb  = lnd.Letterbox(dataset=self)
        rf  = lnd.RandomFlip(flip)
        rc  = lnd.RandomCrop(jitter, True)
        hsv = lnd.HSVShift(hue, saturation, value)
        it  = tf.ToTensor()
        if augment:
            img_tf = tf.Compose([hsv, rc, rf, lb, it])
            anno_tf = tf.Compose([rc, rf, lb])
        else:
            img_tf = tf.Compose([lb, it])
            anno_tf = tf.Compose([lb])
        
        first_img = Image.open(self.img_paths[0])
        w, h = first_img.size
        kwargs = { 'image_width': w, 'image_height':h }

        super(DarknetData, self).__init__(anno_format, self.anno_paths, input_dimension, class_label_map,  identify, img_tf, anno_tf, **kwargs)
