#!/usr/bin/env python
#
#   Test Yolo on VOC
#   Copyright EAVISE
#

import time
import os
import argparse
from statistics import mean
import torch
from torch.autograd import Variable
from torchvision import transforms as tf
from tqdm import tqdm
import cv2
import numpy as np
import brambox.boxes as bbb

import lightnet as ln
ln.log.level = ln.Loglvl.ALL
#ln.log.color = False

# Parameters
WORKERS = 4
PIN_MEM = False
VISDOM = {'server': 'http://localhost', 'port': 8080, 'env': 'YoloVOC Test'}
ROOT = 'data'
CLASS_LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

CLASSES = 20
NETWORK_SIZE = (416, 416)
CONF_THRESH = 0.001
NMS_THRESH = 0.5

BATCH = 64 
BATCH_SUBDIV = 8

assert BATCH % BATCH_SUBDIV == 0, 'Batch subdivision should be a divisor of batch size'


class CustomDataset(ln.data.BramboxData):
    def __init__(self, anno, network):
        def identify(img_id):
            return f'{ROOT}/VOCdevkit/{img_id}'

        lb  = ln.data.Letterbox(self)
        rc  = ln.data.RandomCrop(0, True, 0.1)   # Dont randomcrop image, but crop annos inside of images
        it  = tf.ToTensor()
        img_tf = tf.Compose([rc, lb, it])
        anno_tf = tf.Compose([rc, lb])

        super(CustomDataset, self).__init__('anno_pickle', anno, NETWORK_SIZE, CLASS_LABELS, identify, img_tf, anno_tf)

    def __getitem__(self, index):
        img, anno = super(CustomDataset, self).__getitem__(index)
        for a in anno:
            a.ignore = a.difficult  # Mark difficult annotations as ignore for pr metric
        return img, anno


def test(arguments):
    ln.log(ln.Loglvl.DEBUG, 'Creating network')
    net = ln.models.YoloVoc(CLASSES, arguments.weight, CONF_THRESH, NMS_THRESH)
    net.postprocess = tf.Compose([
        net.postprocess,
        ln.data.TensorToBrambox(NETWORK_SIZE, arguments.names),
    ])
    net.eval()
    if arguments.cuda:
        net.cuda()

    ln.log(ln.Loglvl.DEBUG, 'Creating dataset')
    test = CustomDataset(arguments.test, net)

    if arguments.visdom:
        ln.log(ln.Loglvl.DEBUG, 'Creating visdom visualisation wrapper')
        vis = ln.engine.Visualisation(VISDOM)

    ln.log(ln.Loglvl.DEBUG, 'Running network')
    tot_loss = []
    coord_loss = []
    conf_loss = []
    cls_loss = []
    anno, det = {}, {}
    num_det = 0

    loader = torch.utils.data.DataLoader(
        test,
        batch_size = BATCH // BATCH_SUBDIV,
        shuffle = False,
        drop_last = False,
        num_workers = WORKERS if arguments.cuda else 0,
        pin_memory = PIN_MEM if arguments.cuda else False,
        collate_fn = ln.data.list_collate,
        )
    total = int(len(test) / (BATCH//BATCH_SUBDIV) + .5)
    for data, box in tqdm(loader, total=total):
        if arguments.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)

        output, loss = net(data, box)

        tot_loss.append(net.loss.loss_tot.data[0]*len(box))
        coord_loss.append(net.loss.loss_coord.data[0]*len(box))
        conf_loss.append(net.loss.loss_conf.data[0]*len(box))
        if net.loss.loss_cls is not None:
            cls_loss.append(net.loss.loss_cls.data[0]*len(box))

        key_val = len(anno)
        anno.update({key_val+k: v for k,v in enumerate(target)})
        det.update({key_val+k: v for k,v in enumerate(output)})

    ln.log(ln.Loglvl.DEBUG, 'Computing statistics')
    pr = bbb.pr(det, anno)
    m_ap = round(bbb.ap(*pr)*100, 2)
    tot = round(sum(tot_loss)/len(anno), 5)
    coord = round(sum(coord_loss)/len(anno), 2)
    conf = round(sum(conf_loss)/len(anno), 2)
    if len(cls_loss) > 0:
        cls = round(sum(cls_loss)/len(anno), 2)
        ln.log(ln.Loglvl.VERBOSE, f'mAP:{m_ap}% Loss:{tot} (Coord:{coord} Conf:{conf} Cls:{cls})')
    else:
        ln.log(ln.Loglvl.VERBOSE, f'mAP:{m_ap}% Loss:{tot} (Coord:{coord} Conf:{conf})')

    if arguments.visdom:
        vis.pr(pr, f'pr_{net.seen//BATCH}', title=f'PR - {m_ap}% mAP [{net.seen//BATCH}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a lightnet network')
    parser.add_argument('weight', help='Path to weight file', default=None)
    parser.add_argument('test', help='Pickle annotation file')
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-v', '--visdom', action='store_true', help='Visualize training data with visdom')
    args = parser.parse_args()

    # Parse arguments
    if args.cuda:
        if not torch.cuda.is_available():
            ln.log(ln.Loglvl.ERROR, 'CUDA not available')
            args.cuda = False
        else:
            ln.log(ln.Loglvl.DEBUG, 'CUDA enabled')

    # Test
    test(args)
