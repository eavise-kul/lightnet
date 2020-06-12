#
#   Lightnet postprocessing on Brambox dataframes
#   Copyright EAVISE
#

import logging
import numpy as np
from ..util import BaseTransform

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

__all__ = ['TensorToBrambox', 'ReverseLetterbox']
log = logging.getLogger(__name__)


class TensorToBrambox(BaseTransform):
    """ Converts a tensor to a list of brambox objects.

    Args:
        network_size (tuple): Tuple containing the width and height of the images going in the network
        class_label_map (list, optional): List of class labels to transform the class id's in actual names; Default **None**

    Returns:
        pandas.DataFrame: brambox detection dataframe where the `image` column contains the batch number (int column from 1st tensor column).

    Note:
        Just like everything in PyTorch, this transform only works on batches of images.
        This means you need to wrap your tensor of detections in a list if you want to run this transform on a single image.

    Warning:
        If no `class_label_map` is given, this transform will simply convert the class id's to a string.
    """
    def __init__(self, network_size, class_label_map=None):
        self.width, self.height = network_size
        self.class_label_map = class_label_map
        if self.class_label_map is None:
            log.warning('No class_label_map given. The indexes will be used as class_labels.')

    def __call__(self, boxes):
        if boxes.numel() == 0:
            df = pd.DataFrame(columns=['image', 'class_label', 'id', 'x_top_left', 'y_top_left', 'width', 'height', 'confidence'])
            df.image = df.image.astype(int)
            df.class_label = df.class_label.astype(str)
            return df

        # coords: relative -> absolute
        boxes[:, 1:4:2].mul_(self.width)
        boxes[:, 2:5:2].mul_(self.height)

        # coords: width & height
        boxes[:, 3:5] -= boxes[:, 1:3]

        # Convert to brambox df
        boxes = boxes.cpu().detach().numpy()
        boxes = pd.DataFrame(boxes, columns=['image', 'x_top_left', 'y_top_left', 'width', 'height', 'confidence', 'class_label'])

        # Set column types
        boxes[['image', 'class_label']] = boxes[['image', 'class_label']].astype(int)
        boxes['id'] = np.nan
        boxes[['x_top_left', 'y_top_left', 'width', 'height', 'confidence']] = boxes[['x_top_left', 'y_top_left', 'width', 'height', 'confidence']].astype(float)

        if self.class_label_map is not None:
            boxes.class_label = boxes.class_label.map(dict((i, l) for i, l in enumerate(self.class_label_map)))
        else:
            boxes.class_label = boxes.class_label.astype(str)

        return boxes[['image', 'class_label', 'id', 'x_top_left', 'y_top_left', 'width', 'height', 'confidence']]


class ReverseLetterbox(BaseTransform):
    """ Performs a reverse letterbox operation on the bounding boxes, so they can be visualised on the original image.

    Args:
        network_size (tuple): Tuple containing the width and height of the images going in the network
        image_size (tuple, callable or dict-like): Width and height of the original images (See Note)

    Returns:
        pandas.DataFrame: brambox detection dataframe.

    Note:
        The `image_size` argument can be one of three different types:

        - tuple <width, height> : The same image size will be used for the entire dataframe
        - callable : The argument will be called with the image column name and must return a (width, height) tuple
        - dict-like : This is similar to the callable, but instead of calling the argument, it will use dictionary accessing (self.image_size[img_name])

        Note that if your dimensions are the same for all images, it is faster to pass a tuple,
        as the transformation will be applied to the entire dataframe at once as opposed to grouping it per image and applying the tranform to each group individually.

    Note:
        This transform works on a brambox detection dataframe,
        so you need to apply the :class:`~lightnet.data.TensorToBrambox` transform first.
    """
    def __init__(self, network_size, image_size):
        self.network_size = network_size
        self.image_size = image_size

    def __call__(self, boxes):
        if isinstance(self.image_size, (list, tuple)):
            net_w, net_h = self.network_size[:2]
            im_w, im_h = self.image_size[:2]

            # Get scale and pad
            if im_w == net_w and im_h == net_h:
                scale = 1
            elif im_w / net_w >= im_h / net_h:
                scale = im_w/net_w
            else:
                scale = im_h/net_h
            pad = int((net_w - im_w/scale) / 2), int((net_h - im_h/scale) / 2)

            return self._transform(boxes.copy(), scale, pad)

        return boxes.groupby('image').apply(self._apply_transform)

    def _apply_transform(self, boxes):
        net_w, net_h = self.network_size[:2]
        if callable(self.image_size):
            im_w, im_h = self.image_size(boxes.name)
        else:
            im_w, im_h = self.image_size[boxes.name]

        # Get scale and pad
        if im_w == net_w and im_h == net_h:
            scale = 1
        elif im_w / net_w >= im_h / net_h:
            scale = im_w/net_w
        else:
            scale = im_h/net_h
        pad = int((net_w - im_w/scale) / 2), int((net_h - im_h/scale) / 2)

        # Transform boxes
        return self._transform(boxes.copy(), scale, pad)

    @staticmethod
    def _transform(boxes, scale, pad):
        boxes.x_top_left -= pad[0]
        boxes.y_top_left -= pad[1]

        boxes.x_top_left *= scale
        boxes.y_top_left *= scale
        boxes.width *= scale
        boxes.height *= scale

        return boxes
