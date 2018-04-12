#
#   Lightnet related postprocessing
#   Thers are functions to transform the output of the network to brambox detection objects
#   Copyright EAVISE
#

import logging
import torch
from torch.autograd import Variable
from brambox.boxes.detections.detection import *
from .process import BaseTransform

__all__ = ['GetBoundingBoxes', 'NonMaxSupression', 'TensorToBrambox', 'ReverseLetterbox']
log = logging.getLogger(__name__)


class GetBoundingBoxes(BaseTransform):
    """ Convert output from darknet networks to bounding box tensor.

    Args:
        num_classes (int): number of categories
        anchors (list): 2D list representing anchor boxes (see :class:`lightnet.network.Darknet`)
        conf_thresh (Number [0-1]): Confidence threshold to filter detections

    Returns:
        (list [Batch x Tensor [Boxes x 6]]): **[x_center, y_center, width, height, confidence, class_id]** for every bounding box

    Note:
        The output tensor uses relative values for its coordinates.
    """
    def __init__(self, num_classes, anchors, conf_thresh):
        self.num_classes = num_classes
        self.anchors = anchors
        self.conf_thresh = conf_thresh

        self.mode = 1

    @classmethod
    def apply(cls, network_output, num_classes, anchors, conf_thresh, mode = 1):
        num_anchors = len(anchors)
        anchor_step = len(anchors[0])
        anchors = torch.Tensor(anchors)
        if isinstance(network_output, Variable):
            network_output = network_output.data

        # Check dimensions
        if network_output.dim() == 3:
            network_output.unsqueeze_(0)

        # Variables
        cuda = network_output.is_cuda
        batch = network_output.size(0)
        h = network_output.size(2)
        w = network_output.size(3)

        # Compute xc,yc, w,h, box_score on Tensor
        lin_x = torch.linspace(0, w-1, w).repeat(h,1).view(h*w)
        lin_y = torch.linspace(0, h-1, h).repeat(w,1).t().contiguous().view(h*w)
        anchor_w = anchors[:,0].contiguous().view(1, num_anchors, 1)
        anchor_h = anchors[:,1].contiguous().view(1, num_anchors, 1)
        if cuda:
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        network_output = network_output.view(batch, num_anchors, -1, h*w)  # -1 == 5+num_classes (we can drop feature maps if 1 class)
        network_output[:,:,0,:].sigmoid_().add_(lin_x).div_(w)          # X center
        network_output[:,:,1,:].sigmoid_().add_(lin_y).div_(h)          # Y center
        network_output[:,:,2,:].exp_().mul_(anchor_w).div_(w)           # Width
        network_output[:,:,3,:].exp_().mul_(anchor_h).div_(h)           # Height
        network_output[:,:,4,:].sigmoid_()                              # Box score

        # Compute class_score
        if num_classes > 1:
            cls_scores = torch.nn.functional.softmax(Variable(network_output[:,:,5:,:], volatile=True), 2).data
            cls_max, cls_max_idx = torch.max(cls_scores, 2)
            cls_max.mul_(network_output[:,:,4,:])
        else:
            cls_max = network_output[:,:,4,:]
            cls_max_idx = torch.zeros_like(cls_max)

        if mode == 0:
            network_output = network_output.cpu()
            cls_max = cls_max.cpu()
            cls_max_idx = cls_max_idx.cpu()
            boxes = []
            for b in range(batch):
                box_batch = []
                for a in range(num_anchors):
                    for i in range(h*w):
                        if cls_max[b,a,i] > conf_thresh:
                            box_batch.append([
                                network_output[b,a,0,i],
                                network_output[b,a,1,i],
                                network_output[b,a,2,i],
                                network_output[b,a,3,i],
                                cls_max[b,a,i],
                                cls_max_idx[b,a,i]
                                ])
                boxes.append(torch.Tensor(box_batch))
        else:
            score_thresh = cls_max > conf_thresh
            score_thresh_flat = score_thresh.view(-1)

            if score_thresh.sum() == 0:
                boxes = []
                for i in range(batch):
                    boxes.append(torch.Tensor([]))
                return boxes

            # Mask select boxes > conf_thresh
            coords = network_output.transpose(2, 3)[..., 0:4]
            coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
            scores = cls_max[score_thresh]
            idx = cls_max_idx[score_thresh]
            detections = torch.cat([coords, scores[:, None], idx[:, None]], dim=1)

            # Get indexes of splits between images of batch
            max_det_per_batch = num_anchors * h * w
            slices = [slice(max_det_per_batch * i, max_det_per_batch * (i+1)) for i in range(batch)]
            det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
            split_idx = torch.cumsum(det_per_batch, dim=0)

            # Group detections per image of batch
            boxes = []
            start = 0
            for end in split_idx:
                boxes.append(detections[start: end])
                start = end

        return boxes


class NonMaxSupression(BaseTransform):
    """ Performs nms on the bounding boxes, filtering boxes with a high overlap

    Args:
        nms_thresh (Number [0-1]): Overlapping threshold to filter detections with non-maxima suppresion
        class_nms (Boolean, optional): Whether to perform nms per class; Default **True**
        fast (Boolean, optional): This flag can be used to select a much faster variant on the algorithm, that suppresses slightly more boxes; Default **False**

    Returns:
        (list [Batch x Tensor [Boxes x 6]]): **[x_center, y_center, width, height, confidence, class_id]** for every bounding box

    Note:
        This post-processing function expects the input to be bounding boxes,
        like the ones created by :class:`lightnet.data.GetBoundingBoxes`.  
        Its output is also using relative coordinates.
    """
    def __init__(self, nms_thresh, class_nms = True, fast = False):
        self.nms_thresh = nms_thresh
        self.class_nms = class_nms
        self.fast = fast

        self.mode = 1

    @classmethod
    def apply(cls, boxes, nms_thresh, class_nms = True, fast = False, mode = 1):
        if mode == 0:
            boxes = [cls._nms_old(box, nms_thresh) for box in boxes]
        else:
            boxes = [cls._nms(box, nms_thresh, class_nms, fast) for box in boxes]
        return boxes

    @staticmethod
    def _nms_old(boxes, nms_thresh):
        """ Non maximum suppression.
        Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

        Args:
          boxes (tensor): Bounding boxes from get_detections

        Return:
          (tensor): Pruned boxes
        """
        if boxes.numel() == 0:
            return boxes
        cuda = boxes.is_cuda

        a = boxes[:,:2]
        b = boxes[:,2:4]
        bboxes = torch.cat([a-b/2,a+b/2], 1)
        scores = boxes[:,4]

        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]

        areas = ((x2-x1) * (y2-y1))
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            ids = (iou<=nms_thresh).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids+1]

        keep = torch.LongTensor(keep)
        if cuda:
            keep = keep.cuda()
        return boxes[keep]

    @staticmethod
    def _nms(boxes, nms_thresh, class_nms, fast):
        """ Non maximum suppression.

        Args:
          boxes (tensor): Bounding boxes of one image

        Return:
          (tensor): Pruned boxes
        """
        if boxes.numel() == 0:
            return boxes

        a = boxes[:,:2]
        b = boxes[:,2:4]
        bboxes = torch.cat([a-b/2,a+b/2], 1)
        scores = boxes[:,4]
        classes = boxes[:,5]

        # Sort coordinates by descending score
        scores, order = scores.sort(0, descending=True)
        x1, y1, x2, y2 = bboxes[order].split(1,1)

        # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp_(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp_(min=0)
        
        # Compute iou
        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Filter based on iou (and class)
        conflicting = (ious > nms_thresh).triu(1)

        if class_nms:
            same_class = (classes.unsqueeze(0) == classes.unsqueeze(1))
            conflicting = (conflicting & same_class)

        keep = conflicting.sum(0)
        if not fast:
            l = len(keep) - 1
            for i in range(1, l):
                if keep[i] > 0:
                    keep -= conflicting[i]

        keep = (keep == 0)
        return boxes[order][keep[:,None].expand_as(boxes)].view(-1,6).contiguous()


class TensorToBrambox(BaseTransform):
    """ Converts a tensor to a list of brambox objects. """
    def __init__(self, network_size, class_label_map=None):
        self.network_size = network_size
        self.class_label_map = class_label_map
        if class_label_map is None:
            log.warn('No class_label_map given. The indexes will be used as class_labels.')

    @classmethod
    def apply(cls, boxes, network_size, class_label_map=None):
        if torch.is_tensor(boxes):
            if boxes.dim() == 0:
                return []
            else:
                return cls._convert(boxes, network_size[0], network_size[1], class_label_map)
        else:
            converted_boxes = []
            for box in boxes:
                if box.dim() == 0:
                    converted_boxes.append([])
                else:
                    converted_boxes.append(cls._convert(box, network_size[0], network_size[1], class_label_map))
            return converted_boxes

    @staticmethod
    def _convert(boxes, width, height, class_label_map):
        boxes[:,0:3:2].mul_(width)
        boxes[:,0] -= boxes[:,2] / 2
        boxes[:,1:4:2].mul_(height)
        boxes[:,1] -= boxes[:,3] / 2

        brambox = []
        for box in boxes:
            det = Detection()
            det.x_top_left = box[0]
            det.y_top_left = box[1]
            det.width = box[2]
            det.height = box[3]
            det.confidence = box[4]
            if class_label_map is not None:
                det.class_label = class_label_map[int(box[5])]
            else:
                det.class_label = str(int(box[5]))

            brambox.append(det)

        return brambox


class ReverseLetterbox(BaseTransform):
    """ Performs a reverse letterbox operation on the bounding boxes. """
    def __init__(self, network_size, image_size):
        self.network_size = network_size
        self.image_size = image_size

    @classmethod
    def apply(cls, boxes, network_size, image_size):
        im_w, im_h = image_size[:2]
        net_w, net_h = network_size[:2]

        if im_w == net_w and im_h == net_h:
            scale = 1
        elif im_w / net_w >= im_h / net_h:
            scale = im_w/net_w
        else:
            scale = im_h/net_h
        pad = int((net_w - im_w/scale) / 2), int((net_h - im_h/scale) / 2)

        if isinstance(boxes, Detection):
            return cls._transform([boxes], scale, pad)[0]
        elif len(boxes) == 0:
            return boxes
        elif isinstance(boxes[0], Detection):
            return cls._transform(boxes, scale, pad)
        else:
            converted_boxes = []
            for b in boxes:
                converted_boxes.append(cls._transform(b, scale, pad))
            return converted_boxes

    @staticmethod
    def _transform(boxes, scale, pad):
        for box in boxes:
            box.x_top_left -= pad[0]
            box.y_top_left -= pad[1]

            box.x_top_left *= scale
            box.y_top_left *= scale
            box.width *= scale
            box.height *= scale
        return boxes
