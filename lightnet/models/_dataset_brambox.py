#
#   Lightnet dataset that works with brambox annotations
#   Copyright EAVISE
#

import os
import copy
import logging
from PIL import Image
import lightnet.data as lnd

try:
    import brambox as bb
except ModuleNotFoundError:
    bb = None

__all__ = ['BramboxDataset']
log = logging.getLogger(__name__)


class BramboxDataset(lnd.Dataset):
    """ Dataset for any brambox parsable annotation format.

    Args:
        anno_format (brambox.boxes.formats): Annotation format
        anno_filename (list or str): Annotation filename, list of filenames or expandable sequence
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
        class_label_map (list): List of class_labels
        identify (function, optional): Lambda/function to get image based of annotation filename or image id; Default **replace/add .png extension to filename/id**
        img_transform (torchvision.transforms.Compose): Transforms to perform on the images
        anno_transform (torchvision.transforms.Compose): Transforms to perform on the annotations
        kwargs (dict): Keyword arguments that are passed to the brambox parser
    """
    def __init__(self, anno_format, anno_filename, input_dimension, class_label_map=None, identify=None, img_transform=None, anno_transform=None, **kwargs):
        if bb is None:
            raise ImportError('Brambox needs to be installed for this dataset to work')

        super().__init__(input_dimension)
        self.img_tf = img_transform
        self.anno_tf = anno_transform
        if callable(identify):
            self.id = identify
        else:
            self.id = lambda name: os.path.splitext(name)[0] + '.png'

        # Get annotations
        self.annos = bb.io.load(anno_format, anno_filename, identify=lambda f: f, class_label_map=class_label_map, **kwargs)
        self.keys = self.annos.image.cat.categories

        # Add class_ids
        if class_label_map is None:
            log.warn(f'No class_label_map given, annotations wont have a class_id values for eg. loss function')
        else:
            self.annos['class_id'] = self.annos.class_label.map(dict((l, i) for i, l in enumerate(class_label_map)))

        log.info(f'Dataset loaded: {len(self.keys)} images')

    def __len__(self):
        return len(self.keys)

    @lnd.Dataset.resize_getitem
    def __getitem__(self, index):
        """ Get transformed image and annotations based of the index of ``self.keys``

        Args:
            index (int): index of the ``self.keys`` list containing all the image identifiers of the dataset.

        Returns:
            tuple: (transformed image, list of transformed brambox boxes)
        """
        if index >= len(self):
            raise IndexError(f'list index out of range [{index}/{len(self)-1}]')

        # Load
        img = Image.open(self.id(self.keys[index]))
        anno = bb.util.select_images(self.annos, [self.keys[index]])

        # Transform
        if self.img_tf is not None:
            img = self.img_tf(img)
        if self.anno_tf is not None:
            anno = self.anno_tf(anno)

        return img, anno
