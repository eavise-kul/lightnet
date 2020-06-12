#
#   Lightnet related postprocessing
#   Copyright EAVISE
#

import torch
from ..util import BaseTransform
from distutils.version import LooseVersion

__all__ = ['NonMaxSuppression']

torchversion = LooseVersion(torch.__version__)
version120 = LooseVersion("1.2.0")


class NonMaxSuppression(BaseTransform):
    """ Performs nms on the bounding boxes, filtering boxes with a high overlap.

    Args:
        nms_thresh (Number [0-1]): Overlapping threshold to filter detections with non-maxima suppresion
        class_nms (Boolean, optional): Whether to perform nms per class; Default **True**

    Returns:
        (Tensor [Boxes x 7]]): **[batch_num, x_tl, y_tl, x_br, y_br, confidence, class_id]** for every bounding box

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
        if torchversion >= version120:
            keep = torch.empty(boxes.shape[0], dtype=torch.bool, device=boxes.device)
        else:
            keep = torch.empty(boxes.shape[0], dtype=torch.uint8, device=boxes.device)
        for batch in torch.unique(batches, sorted=False):
            mask = batches == batch
            keep[mask] = self._nms(boxes[mask])

        return boxes[keep]

    def _nms(self, boxes):
        if boxes.numel() == 0:
            return boxes

        bboxes = boxes[:, 1:5]
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
        if torchversion >= version120:
            keep = torch.zeros(len(conflicting), dtype=torch.bool)
        else:
            keep = torch.zeros(len(conflicting), dtype=torch.uint8)
        supress = torch.zeros(len(conflicting), dtype=torch.float)
        for i, row in enumerate(conflicting):
            if not supress[i]:
                keep[i] = 1
                supress[row] = 1

        keep = keep.to(boxes.device)
        return keep.scatter(0, order, keep)
