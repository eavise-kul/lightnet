#
#   Lightnet related postprocessing
#   These are functions to transform the output of the network to brambox detection dataframes
#   Copyright EAVISE
#

import logging
import numpy as np
import torch
from torch.autograd import Variable
from .util import BaseTransform

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

__all__ = ['GetBoundingBoxes', 'NonMaxSuppression', 'NonMaxSupression', 'TensorToBrambox', 'ReverseLetterbox']
log = logging.getLogger(__name__)


class GetBoundingBoxes(BaseTransform):
    """ Convert output from darknet networks to bounding box tensor.

    Args:
        num_classes (int): number of categories
        anchors (list): 2D list representing anchor boxes (see :class:`lightnet.network.Darknet`)
        conf_thresh (Number [0-1]): Confidence threshold to filter detections

    Returns:
        (Tensor [Boxes x 7]]): **[batch_num, x_center, y_center, width, height, confidence, class_id]** for every bounding box

    Note:
        The output tensor uses relative values for its coordinates.
    """
    def __init__(self, num_classes, anchors, conf_thresh):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.anchors = torch.Tensor(anchors)
        self.num_anchors = self.anchors.shape[0]
        self.anchors_step = self.anchors.shape[1]

    def __call__(self, network_output):
        # Check dimensions
        if network_output.dim() == 3:
            network_output.unsqueeze_(0)

        # Variables
        device = network_output.device
        batch = network_output.size(0)
        h = network_output.size(2)
        w = network_output.size(3)

        # Compute xc,yc, w,h, box_score on Tensor
        lin_x = torch.linspace(0, w-1, w).repeat(h, 1).view(h*w).to(device)
        lin_y = torch.linspace(0, h-1, h).view(h, 1).repeat(1, w).view(h*w).to(device)
        anchor_w = self.anchors[:, 0].contiguous().view(1, self.num_anchors, 1).to(device)
        anchor_h = self.anchors[:, 1].contiguous().view(1, self.num_anchors, 1).to(device)

        network_output = network_output.view(batch, self.num_anchors, -1, h*w)  # -1 == 5+num_classes (we can drop feature maps if 1 class)
        network_output[:, :, 0, :].sigmoid_().add_(lin_x).div_(w)               # X center
        network_output[:, :, 1, :].sigmoid_().add_(lin_y).div_(h)               # Y center
        network_output[:, :, 2, :].exp_().mul_(anchor_w).div_(w)                # Width
        network_output[:, :, 3, :].exp_().mul_(anchor_h).div_(h)                # Height
        network_output[:, :, 4, :].sigmoid_()                                   # Box score

        # Compute class_score
        if self.num_classes > 1:
            with torch.no_grad():
                cls_scores = torch.nn.functional.softmax(network_output[:, :, 5:, :], 2)
            cls_max, cls_max_idx = torch.max(cls_scores, 2)
            cls_max_idx = cls_max_idx.float()
            cls_max.mul_(network_output[:, :, 4, :])
        else:
            cls_max = network_output[:, :, 4, :]
            cls_max_idx = torch.zeros_like(cls_max)

        score_thresh = cls_max > self.conf_thresh
        if score_thresh.sum() == 0:
            return torch.tensor([])

        # Mask select boxes > conf_thresh
        coords = network_output.transpose(2, 3)[..., 0:4]
        coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
        scores = cls_max[score_thresh]
        idx = cls_max_idx[score_thresh]

        # Get batch numbers of the detections
        batch_num = score_thresh.view(batch, -1)
        nums = torch.arange(1, batch+1, dtype=batch_num.dtype, device=batch_num.device)
        batch_num = (batch_num * nums[:, None])[batch_num] - 1

        return torch.cat([batch_num[:, None].float(), coords, scores[:, None], idx[:, None]], dim=1)


class NonMaxSuppression(BaseTransform):
    """ Performs nms on the bounding boxes, filtering boxes with a high overlap.

    Args:
        nms_thresh (Number [0-1]): Overlapping threshold to filter detections with non-maxima suppresion
        class_nms (Boolean, optional): Whether to perform nms per class; Default **True**

    Returns:
        (Tensor [Boxes x 7]]): **[batch_num, x_center, y_center, width, height, confidence, class_id]** for every bounding box

    Note:
        This post-processing function expects the input to be bounding boxes,
        like the ones created by :class:`lightnet.data.GetBoundingBoxes` and outputs exactly the same format.
    """
    def __init__(self, nms_thresh, class_nms=True):
        self.nms_thresh = nms_thresh
        self.class_nms = class_nms

    def __call__(self, boxes):
        if boxes.numel() == 0:
            return boxes

        batches = boxes[:, 0]
        keep = torch.empty(boxes.shape[0], dtype=torch.uint8, device=boxes.device)
        for batch in torch.unique(batches, sorted=False):
            mask = batches == batch
            keep[mask] = self._nms(boxes[mask])

        return boxes[keep]

    def _nms(self, boxes):
        if boxes.numel() == 0:
            return boxes

        a = boxes[:, 1:3]
        b = boxes[:, 3:5]
        bboxes = torch.cat([a-b/2, a+b/2], 1)
        scores = boxes[:, 5]
        classes = boxes[:, 6]

        # Sort coordinates by descending score
        scores, order = scores.sort(0, descending=True)
        x1, y1, x2, y2 = bboxes[order].split(1, 1)

        # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)

        # Compute iou
        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Filter based on iou (and class)
        conflicting = (ious > self.nms_thresh).triu(1)

        if self.class_nms:
            classes = classes[order]
            same_class = (classes.unsqueeze(0) == classes.unsqueeze(1))
            conflicting = (conflicting & same_class)

        conflicting = conflicting.cpu()
        keep = torch.zeros(len(conflicting), dtype=torch.uint8)
        supress = torch.zeros(len(conflicting), dtype=torch.float)
        for i, row in enumerate(conflicting):
            if not supress[i]:
                keep[i] = 1
                supress[row] = 1

        keep = keep.to(boxes.device)
        return keep.scatter(0, order, keep)


def NonMaxSupression(*args, **kwargs):
    log.deprecated('NonMaxSupression is deprecated, please use the correctly spelled NonMaxSuppression (2 p\'s)!')
    return NonMaxSuppression(*args, **kwargs)


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

        # X/Y: center -> top_left
        boxes[:, 1:3] -= boxes[:, 3:5] / 2

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
