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
            
            network_output (torch.autograd.Variable): Output tensor from the lightnet network
        """
        boxes = self._get_region_boxes(network_output.data)
        boxes = [self._nms(torch.Tensor(box)) for box in boxes]
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
        '''Non maximum suppression.

        Args:
          boxes (tensor): Bounding boxes from get_detections

        Return:
          (tensor) Pruned boxes
        '''
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


def bbox_to_brambox(boxes, net_size, img_size=None, class_label_map=None):
    """ Convert bounding box array to brambox detection object.
        
    Args:
        boxes (tensor): Detection boxes
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
        det.confidence = box[4]
        if class_label_map is not None:
            det.class_label = class_label_map[int(box[5])]
        else:
            det.class_label = str(int(box[5]))

        det.x_top_left -= pad[0]
        det.y_top_left -= pad[1]
        det.rescale(1/scale)

        dets.append(det)

    return dets
