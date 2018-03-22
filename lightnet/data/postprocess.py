#
#   Lightnet related postprocessing
#   Thers are functions to transform the output of the network to brambox detection objects
#   Copyright EAVISE
#

import logging
import torch
import numpy as np
from torch.autograd import Variable
from brambox.boxes.detections.detection import *

from .process import *

__all__ = ['GetBoundingBoxes', 'TensorToBrambox', 'ReverseLetterbox']
log = logging.getLogger(__name__)


class GetBoundingBoxes:
    """ Convert output from darknet networks to bounding box tensor.
        
    Args:
        network (lightnet.network.Darknet): Network the converter will be used with
        conf_thresh (Number [0-1]): Confidence threshold to filter detections
        nms_thresh(Number [0-1]): Overlapping threshold to filter detections with non-maxima suppresion

    Returns:
        (Batch x Boxes x 6 tensor): **[x_center, y_center, width, height, confidence, class_id]** for every bounding box

    Note:
        The output tensor uses relative values for its coordinates.
    """
    def __init__(self, network, conf_thresh, nms_thresh):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = network.num_classes
        self.anchors = network.anchors
        self.num_anchors = network.num_anchors
        self.anchor_step = len(self.anchors) // self.num_anchors

    def __call__(self, network_output):
        """ Compute bounding boxes after thresholding and nms
            
            network_output (torch.autograd.Variable): Output tensor from the lightnet network
        """
        boxes = self._get_boxes(network_output.data)
        boxes = [self._nms(torch.Tensor(box)) for box in boxes]
        return boxes

    @classmethod
    def apply(cls, network_output, network, conf_thresh, nms_thresh):
        obj = cls(network, conf_thresh, nms_thresh)
        return obj(network_output)

    def _get_boxes(self, output):
        """ Returns array of detections for every image in batch """
        # Check dimensions
        if output.dim() == 3:
            output.unsqueeze_(0)

        # Variables
        cuda = output.is_cuda
        batch = output.size(0)
        h = output.size(2)
        w = output.size(3)

        # Compute xc,yc, w,h, box_score on Tensor
        lin_x = torch.linspace(0, w-1, w).repeat(h,1).view(h*w)
        lin_y = torch.linspace(0, h-1, h).repeat(w,1).t().contiguous().view(h*w)
        anchor_w = torch.Tensor(self.anchors[::2]).view(1, self.num_anchors, 1)
        anchor_h = torch.Tensor(self.anchors[1::2]).view(1, self.num_anchors, 1)
        if cuda:
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        output = output.view(batch, self.num_anchors, -1, h*w)  # -1 == 5+num_classes (we can drop feature maps if 1 class)
        output[:,:,0,:].sigmoid_().add_(lin_x).div_(w)          # X center
        output[:,:,1,:].sigmoid_().add_(lin_y).div_(h)          # Y center
        output[:,:,2,:].exp_().mul_(anchor_w).div_(w)           # Width
        output[:,:,3,:].exp_().mul_(anchor_h).div_(h)           # Height
        output[:,:,4,:].sigmoid_()                              # Box score

        # Compute class_score
        if self.num_classes > 1:
            cls_scores = torch.nn.functional.softmax(Variable(output[:,:,5:,:], volatile=True), 2).data
            cls_max, cls_max_idx = torch.max(cls_scores, 2)
            cls_max.mul_(output[:,:,4,:])
        else:
            cls_max = output[:,:,4,:]
            cls_max_idx = torch.zeros_like(cls_max)

        # Save detection if conf*class_conf is higher than threshold
        output = output.cpu()
        cls_max = cls_max.cpu()
        cls_max_idx = cls_max_idx.cpu()
        boxes = []
        for b in range(batch):
            box_batch = []
            for a in range(self.num_anchors):
                for i in range(h*w):
                    if cls_max[b,a,i] > self.conf_thresh:
                        box_batch.append([
                            output[b,a,0,i],
                            output[b,a,1,i],
                            output[b,a,2,i],
                            output[b,a,3,i],
                            cls_max[b,a,i],
                            cls_max_idx[b,a,i]
                            ])
            boxes.append(box_batch)

        return boxes

    def _nms(self, boxes):
        """ Non maximum suppression.
        Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

        Args:
          boxes (tensor): Bounding boxes from get_detections

        Return:
          (tensor): Pruned boxes
        """

        if boxes.numel() == 0:
            return boxes

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

            ids = (iou<=self.nms_thresh).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids+1]
        return boxes[torch.LongTensor(keep)]


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
