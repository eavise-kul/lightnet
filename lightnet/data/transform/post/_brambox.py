#
#   Lightnet postprocessing brambox
#   Copyright EAVISE
#

import logging
import numpy as np
from ..util import BaseTransform
from ..._imports import pd, bb

__all__ = ['TensorToBrambox']
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
        super().__init__()
        self.class_label_map = class_label_map
        if self.class_label_map is None:
            log.warning('No class_label_map given. The indexes will be used as class_labels.')

    def forward(self, boxes):
        if boxes.numel() == 0:
            df = pd.DataFrame(columns=['image', 'class_label', 'id', 'x_top_left', 'y_top_left', 'width', 'height', 'confidence'])
            df.image = df.image.astype(int)
            df.class_label = df.class_label.astype(str)
            return df

        boxes = boxes.clone()

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
