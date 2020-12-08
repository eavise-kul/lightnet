#
#   Lightnet postprocessing for Anchor based detectors (Darknet)
#   Copyright EAVISE
#

import logging
import torch
from ..util import BaseTransform

__all__ = ['GetDarknetBoxes', 'GetMultiScaleDarknetBoxes', 'GetBoundingBoxes', 'GetMultiScaleBoundingBoxes']
log = logging.getLogger(__name__)


class GetDarknetBoxes(BaseTransform):
    """ Convert output from darknet networks to bounding box tensor.

    Args:
        conf_thresh (Number [0-1]): Confidence threshold to filter detections
        network_stride (Number): Downsampling factor of the network (most lightnet networks have a `stride` attribute)
        anchors (list): 2D list representing anchor boxes (see :class:`lightnet.models.YoloV2`)

    Returns:
        (Tensor [Boxes x 7]]): **[batch_num, x_tl, y_tl, x_br, y_br, confidence, class_id]** for every bounding box
    """
    def __init__(self, conf_thresh, network_stride, anchors):
        super().__init__()
        self.conf_thresh = torch.tensor(conf_thresh)
        self.network_stride = torch.tensor(network_stride)
        self.anchors = torch.tensor(anchors)
        self.num_anchors = torch.tensor(self.anchors.shape[0])
        self.anchors_step = torch.tensor(self.anchors.shape[1])

    def forward(self, network_output):
        device = network_output.device
        batch, channels, h, w = network_output.shape
        num_classes = (channels // self.num_anchors) - 5

        # Compute xc,yc, w,h, box_score on Tensor
        lin_x = torch.linspace(0, w-1, w).repeat(h, 1).view(h*w).to(device)
        lin_y = torch.linspace(0, h-1, h).view(h, 1).repeat(1, w).view(h*w).to(device)
        anchor_w = self.anchors[:, 0].contiguous().view(1, self.num_anchors, 1).to(device)
        anchor_h = self.anchors[:, 1].contiguous().view(1, self.num_anchors, 1).to(device)

        network_output = network_output.view(batch, self.num_anchors, -1, h*w)          # -1 == 5+num_classes (we can drop feature maps if 1 class)
        network_output[:, :, 0, :].sigmoid_().add_(lin_x).mul_(self.network_stride)     # X center
        network_output[:, :, 1, :].sigmoid_().add_(lin_y).mul_(self.network_stride)     # Y center
        network_output[:, :, 2, :].exp_().mul_(anchor_w).mul_(self.network_stride)      # Width
        network_output[:, :, 3, :].exp_().mul_(anchor_h).mul_(self.network_stride)      # Height
        network_output[:, :, 4, :].sigmoid_()                                           # Box score

        # Compute class_score
        if num_classes > 1:
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
            return torch.empty(0, 7, device=device)
        else:
            # Mask select boxes > conf_thresh
            coords = network_output.transpose(2, 3)[..., 0:4]
            coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
            coords = torch.cat([coords[:, 0:2]-coords[:, 2:4]/2, coords[:, 0:2]+coords[:, 2:4]/2], 1)
            scores = cls_max[score_thresh]
            idx = cls_max_idx[score_thresh]

            # Get batch numbers of the detections
            batch_num = score_thresh.view(batch, -1)
            nums = torch.arange(1, batch+1, dtype=torch.uint8, device=batch_num.device)
            batch_num = (batch_num * nums[:, None])[batch_num] - 1

            return torch.cat([batch_num[:, None].float(), coords, scores[:, None], idx[:, None]], dim=1)


def GetBoundingBoxes(*args, **kwargs):
    log.deprecated('GetBoundingBoxes is deprecated, please use the more aptly named "GetDarknetBoxes"')
    log.error('GetDarknetBoxes has different arguments than the previous getBoundingBoxes. Make sure to check the documentation!')
    return GetDarknetBoxes(*args, **kwargs)


class GetMultiScaleDarknetBoxes(GetDarknetBoxes):
    """ Convert the output from multiple yolo output layers (at different scales) to bounding box tensors.

    Args:
        conf_thresh (Number [0-1]): Confidence threshold to filter detections
        network_strides (list): Downsampling factors of the network (most lightnet networks have a `stride` attribute)
        anchors (list): 3D list representing anchor boxes (see :class:`lightnet.models.YoloV3`)

    Returns:
        (Tensor [Boxes x 7]]): **[batch_num, x_tl, y_tl, x_br, y_br, confidence, class_id]** for every bounding box

    Note:
        The `anchors` and `network_strides` should be a list of different values for the different scales.
        When used, this post-processing class calls :class:`~lightnet.data.transform.GetDarknetBoxes` for different scales
        and thus uses different stride and anchors values.

    Warning:
        This post-processing function is not entirely equivalent to the Darknet implementation! |br|
        We just execute the regular :class:`~lightnet.data.transform.GetBoundingBoxes` at multiple scales (different strides and anchors),
        and as such did not implement overlapping class labels.
    """
    def __init__(self, conf_thresh, network_strides, anchors):
        super().__init__(conf_thresh, network_strides[0], anchors[0])
        self.root_strides = network_strides
        self.root_anchors = torch.tensor(anchors, requires_grad=False)

    def forward(self, network_output):
        boxes = []
        for i, output in enumerate(network_output):
            self.network_stride = self.root_strides[i]
            self.anchors = self.root_anchors[i]
            self.num_anchors = self.anchors.shape[0]
            boxes.append(super().forward(output))
        return torch.cat(boxes)


def GetMultiScaleBoundingBoxes(*args, **kwargs):
    log.deprecated('GetMultiScaleBoundingBoxes is deprecated, please use the more aptly named "GetMultiScaleDarknetBoxes"')
    log.error('GetMultiScaleDarknetBoxes has different arguments than the previous getMultiScaleBoundingBoxes. Make sure to check the documentation!')
    return GetMultiScaleDarknetBoxes(*args, **kwargs)
