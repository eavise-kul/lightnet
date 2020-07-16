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

__all__ = ['TensorToBrambox', 'ReverseCrop', 'ReverseLetterbox', 'ReversePad']
log = logging.getLogger(__name__)


class TensorToBrambox(BaseTransform):
    """ Converts a tensor to a list of brambox objects.

    Args:
        class_label_map (list, optional): List of class labels to transform the class id's in actual names; Default **None**

    Returns:
        pandas.DataFrame: brambox detection dataframe where the `image` column contains the batch number (int column from 1st tensor column).

    Note:
        Just like everything in PyTorch, this transform only works on batches of images.
        This means you need to wrap your tensor of detections in a list if you want to run this transform on a single image.

    Warning:
        If no `class_label_map` is given, this transform will simply convert the class id's to a string.
    """
    def __init__(self, class_label_map=None):
        self.class_label_map = class_label_map
        if self.class_label_map is None:
            log.warning('No class_label_map given. The indexes will be used as class_labels.')

    def __call__(self, boxes):
        if boxes.numel() == 0:
            df = pd.DataFrame(columns=['image', 'class_label', 'id', 'x_top_left', 'y_top_left', 'width', 'height', 'confidence'])
            df.image = df.image.astype(int)
            df.class_label = df.class_label.astype(str)
            return df

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


class ReverseCrop(BaseTransform):
    """ Performs a reverse :class:`~lightnet.data.transform.Crop` operation on the bounding boxes, so that the bounding box coordinates are relative to the original image dimensions.

    Args:
        network_size (tuple): Tuple containing the width and height of the images going in the network
        image_size (tuple, callable or dict-like): Width and height of the original images (See Note)

    Returns:
        pandas.DataFrame: brambox dataframe.

    Note:
        The `image_size` argument can be one of three different types:

        - tuple <width, height> : The same image size will be used for the entire dataframe
        - callable : The argument will be called with the image column name and must return a (width, height) tuple
        - dict-like : This is similar to the callable, but instead of calling the argument, it will use dictionary accessing (self.image_size[img_name])

        Note that if your dimensions are the same for all images, it is faster to pass a tuple,
        as the transformation will be applied to the entire dataframe at once as opposed to grouping it per image and applying the tranform to each group individually.

    Warning:
        This post-processing only works when center-cropping images.
        Make sure to set the `center` argument to **True** in your :class:`~lightnet.data.transform.Crop` pre-processing.
    """
    def __init__(self, network_size, image_size):
        self.network_size = network_size
        self.image_size = image_size

    def __call__(self, boxes):
        if isinstance(self.image_size, (list, tuple)):
            net_w, net_h = self.network_size
            im_w, im_h = self.image_size
            scale, pad = self._get_params(net_w, net_h, im_w, im_h)
            return self._transform(boxes.copy(), scale, pad)

        return boxes.groupby('image').apply(self._apply_transform)

    def _apply_transform(self, boxes):
        net_w, net_h = self.network_size
        if callable(self.image_size):
            im_w, im_h = self.image_size(boxes.name)
        else:
            im_w, im_h = self.image_size[boxes.name]

        scale, pad = self._get_params(net_w, net_h, im_w, im_h)
        return self._transform(boxes.copy(), scale, pad)

    @staticmethod
    def _get_params(net_w, net_h, im_w, im_h):
        if net_w / im_w >= net_h / im_h:
            scale = im_w / net_w
            dx = 0
            dy = int(im_h / scale - net_h + 0.5) // 2
        else:
            scale = im_h / net_h
            dx = int(im_w / scale - net_w + 0.5) // 2
            dy = 0

        return scale, (dx, dy)

    @staticmethod
    def _transform(boxes, scale, pad):
        boxes.x_top_left += pad[0]
        boxes.y_top_left += pad[1]

        boxes.x_top_left *= scale
        boxes.y_top_left *= scale
        boxes.width *= scale
        boxes.height *= scale

        return boxes


class ReverseLetterbox(BaseTransform):
    """ Performs a reverse :class:`~lightnet.data.transform.Letterbox` operation on the bounding boxes, so that the bounding box coordinates are relative to the original image dimensions.

    Args:
        network_size (tuple): Tuple containing the width and height of the images going in the network
        image_size (tuple, callable or dict-like): Width and height of the original images (See Note)

    Returns:
        pandas.DataFrame: brambox dataframe.

    Note:
        The `image_size` argument can be one of three different types:

        - tuple <width, height> : The same image size will be used for the entire dataframe
        - callable : The argument will be called with the image column name and must return a (width, height) tuple
        - dict-like : This is similar to the callable, but instead of calling the argument, it will use dictionary accessing (self.image_size[img_name])

        Note that if your dimensions are the same for all images, it is faster to pass a tuple,
        as the transformation will be applied to the entire dataframe at once as opposed to grouping it per image and applying the tranform to each group individually.
    """
    def __init__(self, network_size, image_size):
        self.network_size = network_size
        self.image_size = image_size

    def __call__(self, boxes):
        if isinstance(self.image_size, (list, tuple)):
            net_w, net_h = self.network_size
            im_w, im_h = self.image_size
            scale, pad = self._get_params(net_w, net_h, im_w, im_h)
            return self._transform(boxes.copy(), scale, pad)

        return boxes.groupby('image').apply(self._apply_transform)

    def _apply_transform(self, boxes):
        net_w, net_h = self.network_size
        if callable(self.image_size):
            im_w, im_h = self.image_size(boxes.name)
        else:
            im_w, im_h = self.image_size[boxes.name]

        scale, pad = self._get_params(net_w, net_h, im_w, im_h)
        return self._transform(boxes.copy(), scale, pad)

    @staticmethod
    def _get_params(net_w, net_h, im_w, im_h):
        if im_w == net_w and im_h == net_h:
            scale = 1
        elif im_w / net_w >= im_h / net_h:
            scale = im_w/net_w
        else:
            scale = im_h/net_h
        pad = (net_w - im_w/scale) // 2, (net_h - im_h/scale) // 2

        return scale, pad

    @staticmethod
    def _transform(boxes, scale, pad):
        boxes.x_top_left -= pad[0]
        boxes.y_top_left -= pad[1]

        boxes.x_top_left *= scale
        boxes.y_top_left *= scale
        boxes.width *= scale
        boxes.height *= scale

        return boxes


class ReversePad(BaseTransform):
    """ Performs a reverse :class:`~lightnet.data.transform.Pad` operation on the bounding boxes, so that the bounding box coordinates are relative to the original image dimensions.

    Args:
        network_factor (int or tuple): Tuple containing the factor the width and height need to match
        image_size (tuple, callable or dict-like): Width and height of the original images (See Note)

    Returns:
        pandas.DataFrame: brambox dataframe.

    Note:
        The `image_size` argument can be one of three different types:

        - tuple <width, height> : The same image size will be used for the entire dataframe
        - callable : The argument will be called with the image column name and must return a (width, height) tuple
        - dict-like : This is similar to the callable, but instead of calling the argument, it will use dictionary accessing (self.image_size[img_name])

        Note that if your dimensions are the same for all images, it is faster to pass a tuple,
        as the transformation will be applied to the entire dataframe at once as opposed to grouping it per image and applying the tranform to each group individually.
    """
    def __init__(self, network_factor, image_size):
        self.network_factor = network_factor
        self.image_size = image_size

    def __call__(self, boxes):
        if isinstance(self.image_size, (list, tuple)):
            if isinstance(self.network_factor, int):
                net_w, net_h = self.network_factor, self.network_factor
            else:
                net_w, net_h = self.network_factor
            im_w, im_h = self.image_size

            pad = (net_w - (im_w % net_w)) // 2, (net_h - (im_h % net_h)) // 2
            return self._transform(boxes.copy(), pad)

        return boxes.groupby('image').apply(self._apply_transform)

    def _apply_transform(self, boxes):
        if isinstance(self.network_factor, int):
            net_w, net_h = self.network_factor, self.network_factor
        else:
            net_w, net_h = self.network_factor
        if callable(self.image_size):
            im_w, im_h = self.image_size(boxes.name)
        else:
            im_w, im_h = self.image_size[boxes.name]

        pad = (net_w - (im_w % net_w)) // 2, (net_h - (im_h % net_h)) // 2
        return self._transform(boxes.copy(), pad)

    @staticmethod
    def _transform(boxes, pad):
        boxes.x_top_left -= pad[0]
        boxes.y_top_left -= pad[1]
        return boxes
