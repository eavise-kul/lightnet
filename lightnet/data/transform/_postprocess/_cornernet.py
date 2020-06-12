#
#   Lightnet postprocessing for Anchor based detectors (Darknet)
#   Copyright EAVISE
#

import torch
import torch.nn as nn
from ..util import BaseTransform

__all__ = ['GetCornerBoxes']


class GetCornerBoxes(BaseTransform):
    """ Convert output from cornernet networks to bounding box tensor.

    Args:
        TODO
        #conf_thresh (Number [0-1]): Confidence threshold to filter detections

    Returns:
        (Tensor [Boxes x 7]]): **[batch_num, x_tl, y_tl, x_br, y_br, confidence, class_id]** for every bounding box

    Note:
        The output tensor uses relative values for its coordinates.
    """
    def __init__(self, embedding_thresh, conf_thresh, topk=100, subsample_kernel=0):
        super().__init__()
        self.embedding_thresh = embedding_thresh
        self.conf_thresh = conf_thresh
        self.topk = topk
        self.subsample_kernel = subsample_kernel

    def __call__(self, network_output):
        # Check dimensions
        if network_output.dim() == 3:
            network_output.unsqueeze_(0)

        # Variables
        device = network_output.device
        batch, channels, h, w = network_output.shape
        num_classes = (channels // 2) - 3

        # Split tensor
        network_output = network_output.view(batch, 2, -1, h, w)        # BATCH, TLBR, NUM_CLASSES+3, H, W
        heatmaps = torch.sigmoid(network_output[:, :, :-3])             # BATCH, TLBR, NUM_CLASSES,   H, W
        embedding = network_output[:, :, -3]                            # BATCH, TLBR,                H, W
        offsets = network_output[:, :, -2:]                             # BATCH, TLBR, XY,            H, W

        # Subsample heatmaps
        if self.subsample_kernel:
            maxpool_heat = nn.functional.max_pool2d(heatmaps.view(batch, -1, h, w), self.subsample_kernel, stride=1, padding=(self.subsample_kernel - 1) // 2)
            heatmaps *= maxpool_heat.view(batch, 2, -1, h, w) == heatmaps

        # Get topK corners
        topk_heatmaps, topk_idx = torch.topk(heatmaps.view(batch, 2, -1), self.topk)
        topk_classes = topk_idx // (h * w)
        topk_idx %= (h * w)
        topk_x = (topk_idx % w).float()
        topk_y = (topk_idx // w).float()

        # Add XY offsets
        topk_x += offsets[:, :, 0].reshape(-1)[topk_idx.view(-1)].contiguous().view_as(topk_x)
        topk_y += offsets[:, :, 1].reshape(-1)[topk_idx.view(-1)].contiguous().view_as(topk_y)

        # Combine TL and BR corners
        tl_x = topk_x[:, 0, :, None].expand(-1, self.topk, self.topk)
        tl_y = topk_y[:, 0, :, None].expand(-1, self.topk, self.topk)
        br_x = topk_x[:, 1, None, :].expand(-1, self.topk, self.topk)
        br_y = topk_y[:, 1, None, :].expand(-1, self.topk, self.topk)
        bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=3)

        # Create class filter
        tl_classes = topk_classes[:, 0, :, None].expand(-1, self.topk, self.topk)
        br_classes = topk_classes[:, 1, None, :].expand(-1, self.topk, self.topk)
        class_filter = (tl_classes == br_classes)

        # Create embedding filter
        topk_embed = embedding.reshape(-1)[topk_idx.view(-1)].contiguous().view(batch, 2, -1)
        dist = torch.abs(topk_embed[:, 0, :, None] - topk_embed[:, 1, None, :])
        embedding_filter = dist <= self.embedding_thresh

        # Create corner filter
        corner_filter = (br_x >= tl_x) & (br_y >= tl_y)

        # Create confidence filter
        # NOTE : This is different than the original implementation, where they keep the TOP 1000 detections
        confidence = (topk_heatmaps[:, 0, :, None] + topk_heatmaps[:, 1, None, :]) / 2
        confidence_filter = confidence > self.conf_thresh

        # Get batch number of the detections
        total_filter = class_filter & embedding_filter & corner_filter & confidence_filter
        nums = torch.arange(1, batch+1, dtype=torch.uint8, device=total_filter.device)
        batch_num = total_filter.view(batch, -1)
        batch_num = (batch_num * nums[:, None])[batch_num] - 1

        # Apply filters and combine values
        bboxes = bboxes[total_filter[..., None].expand_as(bboxes)].view(-1, 4)
        confidence = confidence[total_filter].view(-1)
        class_idx = tl_classes[total_filter].view(-1)

        return torch.cat([batch_num[:, None].float(), bboxes, confidence[:, None], class_idx[:, None].float()], dim=1)
