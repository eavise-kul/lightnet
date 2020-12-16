#
#   Darknet YoloV3 RegionLoss
#   Copyright EAVISE
#

import torch
from . import RegionLoss

__all__ = ['MultiScaleRegionLoss']


class MultiScaleRegionLoss(RegionLoss):
    """ Computes region loss from darknet network output and target annotation at multiple scales (yoloV3).

    Args:
        num_classes (int): number of classes to detect
        anchors (list): 3D list representing multiple 2D lists with anchor boxes for the different scales
        stride (list): The different downsampling factors of the different network outputs
        seen (optional, torch.Tensor): How many images the network has already been trained on; Default **0**
        coord_scale (optional, float): weight of bounding box coordinates; Default **1.0**
        noobject_scale (optional, float): weight of regions without target boxes; Default **1.0**
        object_scale (optional, float): weight of regions with target boxes; Default **5.0**
        class_scale (optional, float): weight of categorical predictions; Default **1.0**
        thresh (optional, float): minimum iou between a predicted box and ground truth for them to be considered matching; Default **0.6**
        coord_prefill (optional, int): This parameter controls for how many images the network will prefill the target coordinates, biassing the network to predict the center at **.5,.5**; Default **12800**

    Note:
        All parameters are the same as :class:`~lightnet.network.loss.RegionLoss`, except for `anchors` and `stride`. |br|
        These 2 parameters need separate values for each different network output scale and thus need to be lists of the original parameter.

    Warning:
        This loss function is not entirely equivalent to the Darknet implementation! |br|
        We just execute the regular :class:`~lightnet.network.loss.RegionLoss` at multiple scales (different strides and anchors),
        and as such did not implement overlapping class labels.
    """
    def __init__(self, num_classes, anchors, stride, **kwargs):
        super().__init__(num_classes, anchors[0], stride=stride[0], **kwargs)

        if len(anchors) != len(stride):
            raise IndexError('length of anchors and stride should be equal (number of output scales)')
        self._anchors = torch.tensor(anchors, dtype=torch.float, requires_grad=False)
        self._stride = stride

    def extra_repr(self):
        repr_str = f'classes={self.num_classes}, stride={self.stride}, threshold={self.thresh}, seen={self.seen.item()}\n'
        repr_str += f'coord_scale={self.coord_scale}, object_scale={self.object_scale}, noobject_scale={self.noobject_scale}, class_scale={self.class_scale}\n'
        repr_str += f'anchors='
        start = True
        for anchors in self._anchors:
            if not start:
                repr_str += '| '
            for a in anchors:
                repr_str += f'[{a[0]:.5g}, {a[1]:.5g}] '
            start = False
        return repr_str

    def forward(self, output, target, seen=None):
        device = output[0].device
        loss_total = torch.tensor(0.0, device=device)
        loss_conf = torch.tensor(0.0, device=device)
        loss_coord = torch.tensor(0.0, device=device)
        loss_class = torch.tensor(0.0, device=device)
        if seen is not None:
            self.seen = torch.tensor(seen)

        # Run loss at different scales and sum resulting loss values
        for i, out in enumerate(output):
            self.anchors = self._anchors[i]
            self.num_anchors = self.anchors.shape[0]
            self.anchor_step = self.anchors.shape[1]
            self.stride = self._stride[i]

            super().forward(out, target)
            loss_total += self.loss_total
            loss_conf += self.loss_conf
            loss_coord += self.loss_coord
            loss_class += self.loss_class

        # Overwrite loss values with avg
        self.loss_total = loss_total / len(output)
        self.loss_conf = loss_conf / len(output)
        self.loss_coord = loss_coord / len(output)
        self.loss_class = loss_class / len(output)
        return self.loss_total
