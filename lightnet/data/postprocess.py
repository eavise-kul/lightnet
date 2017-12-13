#
#   Lightnet related postprocessing
#   Thers are functions to transform the output of the network to brambox detection objects
#   Copyright EAVISE
#

import torch
from torch.autograd import Variable
from brambox.boxes.detections.detection import *

from ..logger import *
from ..util import *

__all__ = ['BBoxConverter', 'bbox_to_brambox']


class BBoxConverter:
    """ Convert output from darknet networks to bounding boxes.
        
    Args:
        network (lightnet.network.Darknet): Network the converter will be used with
        conf_thresh (Number [0-1]): Confidence threshold to filter detections
        nms_thresh(Number [0-1]): Overlapping threshold to filter detections with non-maxima suppresion
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
            
            network_output  Output tensor from the lightnet network
        """
        #boxes = self._get_region_boxes_nms(network_output.data)
        all_boxes = self._get_region_boxes(network_output.data)
        boxes = [self._nms(box) for box in all_boxes]
        return boxes

    def _get_region_boxes_nms(self, output):
        """ Returns array of detections for every image in batch """
        # Check dimensions
        if output.dim() == 3:
            output.unsqueeze_(0)

        # Parameters
        cuda = output.is_cuda
        batch = output.size(0)
        h = output.size(2)
        w = output.size(3)

        # Compute xc,yc, w,h, box_score
        lin_x = torch.linspace(0, w-1, w).repeat(h,1).view(h*w)
        lin_y = torch.linspace(0, h-1, h).repeat(w,1).t().contiguous().view(h*w)
        anchor_w = torch.Tensor(self.anchors[::self.anchor_step]).view(self.num_anchors, 1)
        anchor_h = torch.Tensor(self.anchors[1::self.anchor_step]).view(self.num_anchors, 1)
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
            cls_max = cls_max.mul(output[:,:,4,:]).contiguous().view(batch, -1)
        else:
            cls_max = output[:,:,4,:].contiguous().view(batch, -1)
            cls_max_idx = torch.zeros_like(cls_max)

        # Sort class_score for nms
        _, cls_max_sort_id = torch.sort(cls_max, 1, True)

        # Save detection if conf*class_conf is higher than threshold
        output = output.cpu()
        cls_max = cls_max.cpu()
        cls_max_idx = cls_max_idx.view(batch, -1).cpu()
        cls_max_sort_id = cls_max_sort_id.cpu()
        boxes = []
        for b in range(batch):
            box_batch = []
            for i in range(self.num_anchors*h*w):
                idx = cls_max_sort_id[b,i]
                if cls_max[b,idx] < 0:
                    continue
                if cls_max[b,idx] < self.conf_thresh:
                    break

                a = idx // (h*w)
                hw = idx % (h*w)
                box_batch.append([
                    output[b,a,0,hw],
                    output[b,a,1,hw],
                    output[b,a,2,hw],
                    output[b,a,3,hw],
                    cls_max[b,idx],
                    cls_max_idx[b,idx]
                ])

                for j in range(i+1, self.num_anchors*h*w):
                    idx2 = cls_max_sort_id[b,j]
                    a2 = idx2 // (h*w)
                    hw2 = idx2 % (h*w)
                    if cls_max[b,idx2] >= 0 and cls_max[b,idx2] < self.conf_thresh:
                        break
                    if cls_max[b,idx2] >= 0 and bbox_iou(box_batch[-1], output[b,a2,0:4,hw2]) > self.nms_thresh:
                        cls_max[b,idx2] = -1
                        
            boxes.append(box_batch)

        return boxes

    def _get_region_boxes(self, output):
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


def bbox_to_brambox(boxes, net_size, img_size=None, class_label_map=None):
    """ Convert bounding box array to brambox detection object.
        
    Args:
        boxes (list): Detection boxes
        net_size (tuple): Input size of the network (width, height)
        img_size (tuple, optional) Size of the image (width, height); Default **net_size**
        class_label_map (list, optional): class label map to convert class names to an index; Default **None**

    Warning:
        If no class_label_map is given, this function will convert the class_index to a string and use that as class_label.
    """
    net_w, net_h = net_size[:2]
    if img_size is not None:
        im_w, im_h = img_size

        if im_w == net_w and im_h == net_h:
            scale = 1
        elif im_w / net_w >= im_h / net_h:
            scale = net_w/im_w
        else:
            scale = net_h/im_h

        pad = int((net_w-im_w*scale)/2), int((net_h-im_h*scale)/2)
    else:
        scale = 1
        pad = (0,0)

    dets = []
    for box in boxes:
        det = Detection()
        det.x_top_left = (box[0] - box[2]/2) * net_w
        det.y_top_left = (box[1] - box[3]/2) * net_h
        det.width = box[2] * net_w
        det.height = box[3] * net_h
        det.confidence = box[4]*100
        if class_label_map is not None:
            det.class_label = class_label_map[int(box[5])]
        else:
            det.class_label = str(int(box[5]))

        det.x_top_left -= pad[0]
        det.y_top_left -= pad[1]
        det.rescale(1/scale)

        dets.append(det)

    return dets
