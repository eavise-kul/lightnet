#
#   Loss modules
#   Copyright EAVISE
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..util import bbox_iou, bbox_multi_ious
from ..logger import *

__all__ = ['RegionLoss']


class RegionLoss:
    """ Computes region loss from darknet network output and target annotation.
    
    Args:
        network (lightnet.network.Darknet): Network that will be optimised with this loss function
    """
    def __init__(self, network):
        self.net = network
        self.num_classes = network.num_classes
        self.anchors = network.anchors
        self.num_anchors = network.num_anchors
        self.anchor_step = len(self.anchors) // self.num_anchors

        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6

    def __call__(self, output, target):
        """ Compute Region loss.
        
        Args:
            output (torch.autograd.Variable): Output from the network
            target (torch.autograd.Variable or torch.Tensor): Tensor containing the annotation targets (see lightnet.data.AnnoToTensor)
        """
        # Parameters
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cuda = output.is_cuda
        if isinstance(target, Variable):
            target = target.data

        # Get x,y,w,h,conf,cls
        output = output.view(nB, nA, -1, nH*nW)
        x      = output[:,:,0].sigmoid()
        y      = output[:,:,1].sigmoid()
        w      = output[:,:,2]
        h      = output[:,:,3]
        conf   = output[:,:,4].sigmoid()
        if nC > 1:
            cls = output[:,:,5:].contiguous().view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(-1, nC)

        # Create prediction boxes
        pred_boxes = torch.FloatTensor(nB*nA*nH*nW, 4)
        lin_x = torch.linspace(0, nW-1, nW).repeat(nH,1).view(nH*nW)
        lin_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().contiguous().view(nH*nW)
        anchor_w = torch.Tensor(self.anchors[::self.anchor_step]).view(nA, 1)
        anchor_h = torch.Tensor(self.anchors[1::self.anchor_step]).view(nA, 1)
        if cuda:
            pred_boxes = pred_boxes.cuda()
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        pred_boxes[:,0] = (x.data + lin_x).view(-1)
        pred_boxes[:,1] = (y.data + lin_y).view(-1)
        pred_boxes[:,2] = (w.data.exp() * anchor_w).view(-1)
        pred_boxes[:,3] = (h.data.exp() * anchor_h).view(-1)
        pred_boxes = pred_boxes.cpu()

        # Get target values
        coord_mask,conf_mask,cls_mask,tx,ty,tw,th,tconf,tcls = self._build_targets(pred_boxes,target,nH,nW)
        if nC > 1:
            tcls = tcls.view(-1)[cls_mask].long()
            cls_mask = cls_mask.view(-1, 1).repeat(1, nC)

        if cuda:
            tx = tx.cuda()
            ty = ty.cuda()
            tw = tw.cuda()
            th = th.cuda()
            tconf = tconf.cuda()
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()
            if nC > 1:
                tcls = tcls.cuda()
                cls_mask = cls_mask.cuda()

        tx    = Variable(tx, requires_grad=False)
        ty    = Variable(ty, requires_grad=False)
        tw    = Variable(tw, requires_grad=False)
        th    = Variable(th, requires_grad=False)
        tconf = Variable(tconf, requires_grad=False)
        coord_mask = Variable(coord_mask, requires_grad=False)
        conf_mask  = Variable(conf_mask.sqrt(), requires_grad=False)
        if nC > 1:
            tcls  = Variable(tcls, requires_grad=False)
            cls_mask = Variable(cls_mask, requires_grad=False)
            cls      = cls[cls_mask].view(-1, nC)  

        # Compute losses
        mse = nn.MSELoss(size_average=False)
        self.loss_coord  = self.coord_scale * mse(x*coord_mask, tx*coord_mask)/2 \
            + self.coord_scale * mse(y*coord_mask, ty*coord_mask)/2 \
            + self.coord_scale * mse(w*coord_mask, tw*coord_mask)/2 \
            + self.coord_scale * mse(h*coord_mask, th*coord_mask)/2
        self.loss_conf = mse(conf*conf_mask, tconf*conf_mask)/2
        if nC > 1:
            self.loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
            self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls
        else:
            self.loss_cls = None
            self.loss_tot = self.loss_coord + self.loss_conf

        return self.loss_tot

    def _build_targets(self, pred_boxes, target, nH, nW):
        """ Compare prediction boxes and targets, convert targets to network output tensors """
        # Parameters
        nB = target.size(0)
        nT = target.size(1)
        nA = self.num_anchors
        nAnchors = nA*nH*nW
        nPixels  = nH*nW
        seen = self.net.seen + nB

        # Tensors
        conf_mask  = torch.ones(nB, nA, nH*nW) * self.noobject_scale
        coord_mask = torch.zeros(nB, nA, nH*nW)
        cls_mask   = torch.zeros(nB, nA, nH*nW)
        tx         = torch.zeros(nB, nA, nH*nW) 
        ty         = torch.zeros(nB, nA, nH*nW) 
        tw         = torch.zeros(nB, nA, nH*nW) 
        th         = torch.zeros(nB, nA, nH*nW) 
        tconf      = torch.zeros(nB, nA, nH*nW)
        tcls       = torch.zeros(nB, nA, nH*nW)

        if seen < 12800:
            coord_mask.fill_(1)
            if self.anchor_step == 4:
                tx = torch.Tensor(self.anchors[2::self.anchor_step]).view(1,nA,1).repeat(nB,1,nH*nW)
                ty = torch.Tensor(self.anchors[3::self.anchor_step]).view(1,nA,1).repeat(nB,1,nH*nW)
            else:
                tx.fill_(0.5)
                ty.fill_(0.5)

        # Set conf_mask to 0 if iou > thresh
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors]
            cur_ious = torch.zeros(nAnchors)
            for t in range(nT):
                if target[b][t][0] < 0:
                    break
                gx = target[b][t][1] * nW
                gy = target[b][t][2] * nH
                gw = target[b][t][3] * nW
                gh = target[b][t][4] * nH
                cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1)
                cur_ious = torch.max(cur_ious, bbox_multi_ious(cur_pred_boxes, cur_gt_boxes))
            conf_mask[b][cur_ious>self.thresh] = 0

        # Loop over targets and construct tensors
        for b in range(nB):
            for t in range(nT):
                if target[b][t][0] < 0:
                    break
                best_iou = 0.0
                best_n = -1
                min_dist = 10000
                gx = target[b][t][1] * nW
                gy = target[b][t][2] * nH
                gw = target[b][t][3] * nW
                gh = target[b][t][4] * nH
                gi = int(gx)
                gj = int(gy)
                gt_box = [0, 0, gw, gh]
                for n in range(nA):
                    aw = self.anchors[self.anchor_step*n]
                    ah = self.anchors[self.anchor_step*n+1]
                    anchor_box = [0, 0, aw, ah]
                    iou  = bbox_iou(anchor_box, gt_box)
                    if self.anchor_step == 4:
                        ax = self.anchors[self.anchor_step*n+2]
                        ay = self.anchors[self.anchor_step*n+3]
                        dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                    if iou > best_iou:
                        best_iou = iou
                        best_n = n
                    elif self.anchor_step==4 and iou == best_iou and dist < min_dist:
                        best_iou = iou
                        best_n = n
                        min_dist = dist

                gt_box = [gx, gy, gw, gh]
                pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
                iou = bbox_iou(gt_box, pred_box)

                coord_mask[b][best_n][gj*nW+gi] = 1
                cls_mask[b][best_n][gj*nW+gi] = 1
                conf_mask[b][best_n][gj*nW+gi] = self.object_scale
                tx[b][best_n][gj*nW+gi] = gx - gi
                ty[b][best_n][gj*nW+gi] = gy - gj
                tw[b][best_n][gj*nW+gi] = math.log(gw/self.anchors[self.anchor_step*best_n])
                th[b][best_n][gj*nW+gi] = math.log(gh/self.anchors[self.anchor_step*best_n+1])
                tconf[b][best_n][gj*nW+gi] = iou
                tcls[b][best_n][gj*nW+gi] = target[b][t][0]

        return coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls
