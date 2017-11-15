#
#   Bounding box computations
#   Copyright EAVISE
#

import torch
from torch.autograd import Variable

from .logger import *

__all__ = ['BBoxConverter']


class BBoxConverter:
    """ Convert output from darknet networks to bounding boxes """
    def __init__(self, conf_thresh, nms_thresh, anchors, num_classes=1):
        super(BBoxConverter, self).__init__()
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchors = anchors
        self.num_classes = num_classes
        self.num_anchors = len(self.anchors) // 2

    def __call__(self, network_output):
        all_boxes = self._get_region_boxes(network_output.data)
        boxes = [self._nms(box) for box in all_boxes]
        return boxes

    def _get_region_boxes(self, output):
        """ Returns array of detections for every image in batch """
        # Check dimensions
        if output.dim() == 3:
            output.unsqueeze_(0)
        if output.size(1) != (5+self.num_classes)*self.num_anchors:
            log(Loglvl.ERROR, 'Output not correct with number of anchors and classes', ValueError)

        # Variables
        cuda = output.is_cuda
        batch = output.size(0)
        h = output.size(2)
        w = output.size(3)

        # Compute xc,yc, w,h, conf,class_conf on Tensor
        lin_x = torch.linspace(0, w-1, w).repeat(h,1).view(h*w)
        lin_y = torch.linspace(0, h-1, h).repeat(w,1).t().contiguous().view(h*w)
        anchor_w = torch.Tensor(self.anchors[::2]).view(1, self.num_anchors, 1)
        anchor_h = torch.Tensor(self.anchors[1::2]).view(1, self.num_anchors, 1)
        if cuda:
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        output = output.view(batch, self.num_anchors, -1, h*w)  # -1 == 5+num_classes -> if only 1 class we could drop that 1 extra feature map 
        output[:,:,0,:].sigmoid_().add_(lin_x).div_(w)          # X center
        output[:,:,1,:].sigmoid_().add_(lin_y).div_(h)          # Y center
        output[:,:,2,:].exp_().mul_(anchor_w).div_(w)           # Width
        output[:,:,3,:].exp_().mul_(anchor_h).div_(h)           # Height
        output[:,:,4,:].sigmoid_()                              # Box score

        # Compute class scores
        if num_classes > 1:
            cls_scores = torch.nn.functional.softmax(Variable(output[:,:,5:,:]), 2).data
            cls_max, cls_max_idx = torch.max(cls_scores, 2)
            cls_max.mul_(output[:,:,4,:])
        else:
            cls_max = output[:,:,4,:]
            cls_max_idx = torch.zeros_like(cls_max)

        # Save detection if conf*class_conf is higher than threshold
        output = output.cpu()
        cls_max = cls_max.cpu()
        boxes = []
        for b in range(batch):
            box_batch = []
            for a in range(self.num_anchors):
                for i in range(h*w):
                    if cls_max[b,a,i] > self.conf_thresh:
                        box_batch.append([output[b,a,0,i], output[b,a,1,i], output[b,a,2,i], output[b,a,3,i], cls_max[b,a,i], cls_max_idx[b,a,i]])
            boxes.append(box_batch)

        return boxes

    def _nms(self, detections):
        """ Returns pruned detections after nms """
        det_len = len(detections)

        if len(detections) == 0:
            return detections

        det_scores = torch.Tensor([det[4] for det in detections])
        _,sortIds = torch.sort(det_scores, 0, True)
        out_detections = []
        for i in range(det_len):
            det_i = detections[sortIds[i]]
            if det_i[4] > 0:
                out_detections.append(det_i)
                for j in range(i+1, det_len):
                    det_j = detections[sortIds[j]]
                    if det_j[4] > 0 and bbox_iou(det_i, det_j) > self.nms_thresh:
                        det_j[4] = 0

        return out_detections


def bbox_iou(box1, box2):
    """ Compute IOU between 2 bounding boxes
        Box format: [xc, yc, w, h]
    """
    mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
    Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
    my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
    My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
    w1 = box1[2]
    h1 = box1[3]
    w2 = box2[2]
    h2 = box2[3]

    uw = Mx - mx
    uh = My - my
    iw = w1 + w2 - uw
    ih = h1 + h2 - uh
    if iw <= 0 or ih <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    iarea = iw * ih
    uarea = area1 + area2 - iarea
    return iarea/uarea
