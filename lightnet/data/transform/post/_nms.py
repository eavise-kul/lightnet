#
#   Lightnet related postprocessing
#   Copyright EAVISE
#

import logging
import numpy as np
import torch
from ..util import BaseTransform
from ..._imports import bb


__all__ = ['NMS', 'NMSFast', 'NMSSoft', 'NMSSoftFast', 'NonMaxSuppression']
log = logging.getLogger(__name__)


class NMS(BaseTransform):
    """ Performs non-maximal suppression on the bounding boxes, filtering boxes with a high overlap.
    This function can either work on a pytorch bounding box tensor or a brambox dataframe.

    Args:
        nms_thresh (Number [0-1]): Overlapping threshold to filter detections with non-maxima suppresion
        class_nms (Boolean, optional): Whether to perform nms per class; Default **True**
        force_cpu (Boolean, optional): Whether to force a part of the computation on CPU (tensor only, see Note); Default **True**
        reset_index (Boolean, optional): Whether to reset the index of the returned dataframe (dataframe only); Default **True**

    Input:
        boxes (Tensor [Boxes x 7] or pandas.Dataframe): bounding boxes

    Returns:
        boxes (Tensor [Boxes x 7] or pandas.Dataframe): filtered bounding boxes

    Note:
        This post-processing function expects the input bounding boxes to be either a PyTorch tensor or a brambox dataframe.

        The PyTorch tensor needs to be formatted as follows: **[batch_num, x_tl, y_tl, x_br, y_br, confidence, class_id]** for every bounding box.
        This corresponds to the output from the Get***Boxes classes available in lightnet.

        The brambox dataframe should be a detection dataframe,
        as it needs the x_top_left, y_top_left, width, height, confidence and class_label columns.

    Note:
        A part of the computation of NMS involve a sequential loop through the boxes.
        This is quite difficult to implement efficiently on the GPU, using the PyTorch API.
        We thus concluded empirically that it is usually more efficient to move GPU tensors to the CPU for this part of the computations. |br|
        By passing false to `force_cpu`, you can disable this behavior and perform all the computations on the original device of the input tensor.
    """
    def __init__(self, nms_thresh, class_nms=True, force_cpu=True, reset_index=True):
        super().__init__()
        self.nms_thresh = nms_thresh
        self.class_nms = class_nms
        self.force_cpu = force_cpu
        self.reset_index = reset_index

    def forward(self, boxes):
        if isinstance(boxes, torch.Tensor):
            return self._torch(boxes)
        else:
            return self._pandas(boxes)

    def _torch(self, boxes):
        if boxes.numel() == 0:
            return boxes

        batches = boxes[:, 0]
        keep = torch.empty(boxes.shape[0], dtype=torch.bool, device=boxes.device)
        for batch in torch.unique(batches, sorted=False):
            mask = batches == batch
            keep[mask] = self._torch_nms(boxes[mask])

        return boxes[keep]

    def _torch_nms(self, boxes):
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

        if self.force_cpu:
            conflicting = conflicting.cpu()
        keep = torch.zeros(conflicting.shape[0], dtype=torch.bool, device=conflicting.device)
        supress = torch.zeros(conflicting.shape[0], dtype=torch.bool, device=conflicting.device)
        for i, row in enumerate(conflicting):
            if not supress[i]:
                keep[i] = True
                supress[row] = True

        keep = keep.to(boxes.device)
        return keep.scatter(0, order, keep)

    @torch.jit.ignore
    def _pandas(self, boxes):
        if len(boxes.index) == 0:
            return boxes
        boxes = boxes.groupby('image', group_keys=False, observed=True).apply(self._pandas_nms)
        if self.reset_index:
            return boxes.reset_index(drop=True)
        return boxes

    def _pandas_nms(self, boxes):
        boxes = boxes.sort_values('confidence', ascending=False)
        ious = bb.stat.coordinates.iou(boxes, boxes, bias=0)
        scores = boxes['confidence'].values

        # Filter based on iou (and class)
        conflicting = np.triu(ious > self.nms_thresh, 1)
        if self.class_nms:
            classes = boxes['class_label'].values
            same_class = (classes[None, ...] == classes[..., None])
            conflicting = (conflicting & same_class)

        keep = np.zeros(conflicting.shape[0], dtype=np.bool)
        supress = np.zeros(conflicting.shape[0], dtype=np.bool)
        for i, row in enumerate(conflicting):
            if not supress[i]:
                keep[i] = True
                supress[row] = True

        return boxes[keep]


class NMSFast(NMS):
    """ Performs fast NMS on the bounding boxes, filtering boxes with a high overlap.
    This function can either work on a pytorch bounding box tensor or a brambox dataframe.

    This faster alternative makes a small "mistake" during NMS computation,
    in order to remove a necessary loop in the code, allowing it to run faster.
    The speed increase should be mostly notable when performing NMS with PyTorch tensors on the GPU.

    The difference is explained in the image below, where the boxes A and B overlap enough to be filtered out
    and the boxes B and C as well (but A and C do not). |br|
    Regular NMS will keep box C in this situation, because box B gets filtered out and is thus not there to remove C.
    Fast NMS will not do this and will only keep box A in this situation. |br|
    Depending on the use-case (closely clustered and overlapping objects), this might be a problem or not.

    .. figure:: /.static/api/nms-fast.*
       :width: 100%
       :alt: Fast NMS problem

       Regular NMS will keep both boxes A and C, but Fast NMS will only keep A in this example.

    Args:
        nms_thresh (Number [0-1]): Overlapping threshold to filter detections with non-maxima suppresion
        class_nms (Boolean, optional): Whether to perform nms per class; Default **True**
        reset_index (Boolean, optional): Whether to reset the index of the returned dataframe (dataframe only); Default **True**

    Input:
        boxes (Tensor [Boxes x 7] or pandas.Dataframe): bounding boxes

    Returns:
        boxes (Tensor [Boxes x 7] or pandas.Dataframe): filtered bounding boxes

    Note:
        This post-processing function expects the input bounding boxes to be either a PyTorch tensor or a brambox dataframe.

        The PyTorch tensor needs to be formatted as follows: **[batch_num, x_tl, y_tl, x_br, y_br, confidence, class_id]** for every bounding box.
        This corresponds to the output from the Get***Boxes classes available in lightnet.

        The brambox dataframe should be a detection dataframe,
        as it needs the x_top_left, y_top_left, width, height, confidence and class_label columns.
    """
    def __init__(self, nms_thresh, class_nms=True, reset_index=True):
        super().__init__(nms_thresh, class_nms, False, reset_index)

    def _torch_nms(self, boxes):
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

        keep = conflicting.sum(0) == 0
        return keep.scatter(0, order, keep)

    @torch.jit.ignore
    def _pandas_nms(self, boxes):
        boxes = boxes.sort_values('confidence', ascending=False)
        ious = bb.stat.coordinates.iou(boxes, boxes, bias=0)
        scores = boxes['confidence'].values

        conflicting = np.triu(ious > self.nms_thresh, 1)
        if self.class_nms:
            classes = boxes['class_label'].values
            same_class = (classes[None, ...] == classes[..., None])
            conflicting = (conflicting & same_class)

        # Return filtered boxes
        keep = conflicting.sum(0) == 0
        return boxes[keep]


class NMSSoft(BaseTransform):
    """ Performs soft NMS with exponential decaying on the bounding boxes, as explained in :cite:`soft_nms`.
    This function can either work on a pytorch bounding box tensor or a brambox dataframe.

    Args:
        sigma (Number): Sensitivity value for the confidence rescaling (exponential decay)
        conf_thresh (Number [0-1], optional): Confidence threshold to filter the bounding boxes after decaying them; Default **0**
        class_nms (Boolean, optional): Whether to perform nms per class; Default **True**
        force_cpu (Boolean, optional): Whether to force a part of the computation on CPU (tensor only, see Note); Default **True**
        reset_index (Boolean, optional): Whether to reset the index of the returned dataframe (dataframe only); Default **True**

    Input:
        boxes (Tensor [Boxes x 7] or pandas.Dataframe): bounding boxes

    Returns:
        boxes (Tensor [Boxes x 7] or pandas.Dataframe): filtered bounding boxes

    Note:
        This post-processing function expects the input bounding boxes to be either a PyTorch tensor or a brambox dataframe.

        The PyTorch tensor needs to be formatted as follows: **[batch_num, x_tl, y_tl, x_br, y_br, confidence, class_id]** for every bounding box.
        This corresponds to the output from the Get***Boxes classes available in lightnet.

        The brambox dataframe should be a detection dataframe,
        as it needs the x_top_left, y_top_left, width, height, confidence and class_label columns.
    """
    def __init__(self, sigma, conf_thresh=0, class_nms=True, force_cpu=True, reset_index=True):
        super().__init__()
        self.sigma = sigma
        self.conf_thresh = conf_thresh
        self.class_nms = class_nms
        self.force_cpu = force_cpu
        self.reset_index = reset_index

    def forward(self, boxes):
        if isinstance(boxes, torch.Tensor):
            return self._torch(boxes)
        else:
            return self._pandas(boxes)

    def _torch(self, boxes):
        if boxes.numel() == 0:
            return boxes

        batches = boxes[:, 0]
        for batch in torch.unique(batches, sorted=False):
            mask = batches == batch
            boxes[mask, 5] = self._torch_nms(boxes[mask])

        if self.conf_thresh > 0:
            keep = boxes[:, 5] > self.conf_thresh
            return boxes[keep]

        return boxes

    def _torch_nms(self, boxes):
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

        # Filter class
        if self.class_nms:
            classes = classes[order]
            same_class = (classes.unsqueeze(0) == classes.unsqueeze(1))
            ious *= same_class

        # Decay scores
        decay = torch.exp(-(ious ** 2) / self.sigma)
        if self.force_cpu:
            scores = scores.cpu()
            order = order.cpu()
            decay = decay.cpu()

        tempscores = scores.clone()
        for _ in range(scores.shape[0]):
            maxidx = tempscores.argmax()
            maxscore = tempscores[maxidx]
            if maxscore <= self.conf_thresh:
                break

            tempscores[maxidx] = -1
            mask = tempscores != -1
            tempscores[mask] *= decay[maxidx, mask]
            scores[mask] = tempscores[mask]

        scores = scores.to(boxes.device)
        order = order.to(boxes.device)
        return scores.scatter(0, order, scores)

    @torch.jit.ignore
    def _pandas(self, boxes):
        if len(boxes.index) == 0:
            return boxes

        boxes = boxes.groupby('image', group_keys=False, observed=True).apply(self._pandas_nms)
        if self.conf_thresh > 0:
            boxes = boxes[boxes.confidence > self.conf_thresh].copy()
        if self.reset_index:
            return boxes.reset_index(drop=True)

        return boxes

    def _pandas_nms(self, boxes):
        boxes = boxes.sort_values('confidence', ascending=False)
        scores = boxes['confidence'].values
        ious = bb.stat.coordinates.iou(boxes, boxes, bias=0)

        # Filter class
        if self.class_nms:
            classes = boxes['class_label'].values
            same_class = (classes[None, ...] == classes[..., None])
            ious *= same_class

        # Decay scores
        decay = np.exp(-(ious ** 2) / self.sigma)
        tempscores = scores.copy()
        for _ in range(scores.shape[0]):
            maxidx = tempscores.argmax()
            maxscore = tempscores[maxidx]
            if maxscore <= self.conf_thresh:
                break

            tempscores[maxidx] = -1
            mask = tempscores != -1
            tempscores[mask] *= decay[maxidx, mask]
            scores[mask] = tempscores[mask]

        # Set scores back
        boxes['confidence'] = scores
        return boxes


class NMSSoftFast(BaseTransform):
    """ Faster version of SoftNMS which filters boxes with a high overlap, using exponential decay.
    This function can either work on a pytorch bounding box tensor or a brambox dataframe.

    This version of NMS makes the same "mistake" as :class:`~lightnet.data.transform.NMSFast`,
    which in turn allows it to be faster than the regular :class:`~lightnet.data.transform.NMSSoft` algorithm.
    The speed increase should be mostly notable when performing NMS with PyTorch tensors on the GPU.

    Args:
        sigma (Number): Sensitivity value for the confidence rescaling (exponential decay)
        conf_thresh (Number [0-1], optional): Confidence threshold to filter the bounding boxes after decaying them; Default **0**
        class_nms (Boolean, optional): Whether to perform nms per class; Default **True**
        reset_index (Boolean, optional): Whether to reset the index of the returned dataframe (dataframe only); Default **True**

    Input:
        boxes (Tensor [Boxes x 7] or pandas.Dataframe): bounding boxes

    Returns:
        boxes (Tensor [Boxes x 7] or pandas.Dataframe): filtered bounding boxes

    Note:
        This post-processing function expects the input bounding boxes to be either a PyTorch tensor or a brambox dataframe.

        The PyTorch tensor needs to be formatted as follows: **[batch_num, x_tl, y_tl, x_br, y_br, confidence, class_id]** for every bounding box.
        This corresponds to the output from the Get***Boxes classes available in lightnet.

        The brambox dataframe should be a detection dataframe,
        as it needs the x_top_left, y_top_left, width, height, confidence and class_label columns.
    """
    def __init__(self, sigma, conf_thresh=0, class_nms=True, reset_index=True):
        super().__init__()
        self.sigma = sigma
        self.conf_thresh = conf_thresh
        self.class_nms = class_nms
        self.reset_index = reset_index

    def forward(self, boxes):
        if isinstance(boxes, torch.Tensor):
            return self._torch(boxes)
        else:
            return self._pandas(boxes)

    def _torch(self, boxes):
        if boxes.numel() == 0:
            return boxes

        batches = boxes[:, 0]
        for batch in torch.unique(batches, sorted=False):
            mask = batches == batch
            boxes[mask, 5] = self._torch_nms(boxes[mask])

        if self.conf_thresh > 0:
            keep = boxes[:, 5] > self.conf_thresh
            return boxes[keep]

        return boxes

    def _torch_nms(self, boxes):
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

        # Filter class
        if self.class_nms:
            classes = classes[order]
            same_class = (classes.unsqueeze(0) == classes.unsqueeze(1))
            ious *= same_class

        # Decay scores
        decay = ious.triu(1)
        decay = torch.exp(-(decay ** 2) / self.sigma).prod(0)
        scores *= decay
        return scores.scatter(0, order, scores)

    @torch.jit.ignore
    def _pandas(self, boxes):
        if len(boxes.index) == 0:
            return boxes

        boxes = boxes.groupby('image', group_keys=False, observed=True).apply(self._pandas_nms)
        if self.conf_thresh > 0:
            boxes = boxes[boxes.confidence > self.conf_thresh].copy()
        if self.reset_index:
            return boxes.reset_index(drop=True)

        return boxes

    def _pandas_nms(self, boxes):
        boxes = boxes.sort_values('confidence', ascending=False)
        scores = boxes['confidence'].values
        ious = bb.stat.coordinates.iou(boxes, boxes, bias=0)

        # Filter class
        if self.class_nms:
            classes = boxes['class_label'].values
            same_class = (classes[None, ...] == classes[..., None])
            ious *= same_class

        # Decay scores
        decay = np.triu(ious, 1)
        decay = np.prod(np.exp(-(decay ** 2) / self.sigma), 0)
        scores *= decay

        # Set scores back
        boxes['confidence'] = scores
        return boxes


def NonMaxSuppression(*args, **kwargs):
    log.deprecated('NonMaxSuppression is deprecated, please use "NMS"')
    return NMS(*args, **kwargs)
