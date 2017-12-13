#
#   Lightnet util: Bits and bops needed in multiple subpackages of lightnet
#   Copyright EAVISE
#

import torch

__all__ = ['bbox_iou', 'bbox_multi_ious']


def bbox_iou(box1, box2):
    """ Compute IOU between 2 bounding boxes
        Box format: [xc, yc, w, h]
    """
    mx = min(box1[0]-box1[2]/2, box2[0]-box2[2]/2)
    Mx = max(box1[0]+box1[2]/2, box2[0]+box2[2]/2)
    my = min(box1[1]-box1[3]/2, box2[1]-box2[3]/2)
    My = max(box1[1]+box1[3]/2, box2[1]+box2[3]/2)
    w1 = box1[2]
    h1 = box1[3]
    w2 = box2[2]
    h2 = box2[3]

    uw = Mx - mx
    uh = My - my
    iw = w1 + w2 - uw
    ih = h1 + h2 - uh
    if iw <= 0 or ih <= 0:
        return 0

    area1 = w1 * h1
    area2 = w2 * h2
    iarea = iw * ih
    uarea = area1 + area2 - iarea
    return iarea/uarea


def bbox_multi_ious(boxes1, boxes2):
    """ Compute IOU between 2 lists of bounding boxes
        List format: [[xc, yc, w, h],...]
    """
    mx = torch.min(boxes1[:,0]-boxes1[:,2]/2, boxes2[:,0]-boxes2[:,2]/2)
    Mx = torch.max(boxes1[:,0]+boxes1[:,2]/2, boxes2[:,0]+boxes2[:,2]/2)
    my = torch.min(boxes1[:,1]-boxes1[:,3]/2, boxes2[:,1]-boxes2[:,3]/2)
    My = torch.max(boxes1[:,1]+boxes1[:,3]/2, boxes2[:,1]+boxes2[:,3]/2)
    w1 = boxes1[:,2]
    h1 = boxes1[:,3]
    w2 = boxes2[:,2]
    h2 = boxes2[:,3]

    uw = Mx - mx
    uh = My - my
    iw = w1 + w2 - uw
    ih = h1 + h2 - uh

    area1 = w1 * h1
    area2 = w2 * h2
    iarea = iw * ih
    iarea[(iw <= 0) + (ih <= 0) > 0] = 0
    uarea = area1 + area2 - iarea
    return iarea/uarea
