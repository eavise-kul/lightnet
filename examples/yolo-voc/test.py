#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Test Yolo on Pascal VOC
#

import os
import argparse
import logging
from pathlib import Path
from statistics import mean
import numpy as np
from tqdm import tqdm
import visdom
import torch
from torchvision import transforms as tf
import brambox.boxes as bbb
import lightnet as ln

log = logging.getLogger('lightnet.test')
ln.logger.setLogFile('best_pr.log', filemode='a')           # Enable logging of test logs (By appending, multiple runs will keep writing to same file, allowing to search the best)
#ln.logger.setConsoleLevel(logging.NOTSET)                  # Enable debug prints in terminal
#ln.logger.setConsoleColor(False)                           # Disable colored terminal output

# Parameters
WORKERS = 8
PIN_MEM = True
ROOT = 'data'
TESTFILE = f'{ROOT}/test.pkl'

CLASSES = 20
LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
NETWORK_SIZE = (416, 416)
CONF_THRESH = 0.001
NMS_THRESH = 0.5

BATCH = 64
MINI_BATCH = 8


class CustomDataset(ln.data.BramboxData):
    def __init__(self, anno, network):
        def identify(img_id):
            return f'{ROOT}/VOCdevkit/{img_id}.jpg'

        lb  = ln.data.transform.Letterbox(NETWORK_SIZE)
        it  = tf.ToTensor()
        img_tf = ln.data.transform.Compose([lb, it])
        anno_tf = ln.data.transform.Compose([lb])

        super(CustomDataset, self).__init__('anno_pickle', anno, NETWORK_SIZE, LABELS, identify, img_tf, anno_tf)

    def __getitem__(self, index):
        img, anno = super(CustomDataset, self).__getitem__(index)
        for a in anno:
            a.ignore = a.difficult  # Mark difficult annotations as ignore for pr metric
        return img, anno


def test(arguments):
    log.debug('Creating network')
    net = ln.models.Yolo(CLASSES, arguments.weight, CONF_THRESH, NMS_THRESH)
    net.postprocess.append(ln.data.transform.TensorToBrambox(NETWORK_SIZE, LABELS))
    net.eval()
    if arguments.cuda:
        net.cuda()

    log.debug('Creating dataset')
    loader = torch.utils.data.DataLoader(
        CustomDataset(TESTFILE, net),
        batch_size = MINI_BATCH,
        shuffle = False,
        drop_last = False,
        num_workers = WORKERS if arguments.cuda else 0,
        pin_memory = PIN_MEM if arguments.cuda else False,
        collate_fn = ln.data.list_collate,
    )

    if arguments.visdom:
        log.debug('Creating visdom visualisation wrappers')
        vis = visdom.Visdom(port=8080)
        plot_pr = ln.engine.LinePlotter(vis, 'pr', opts=dict(xlabel='Recall', ylabel='Precision', title='Precision Recall', xtickmin=0, xtickmax=1, ytickmin=0, ytickmax=1, showlegend=True))

    log.debug('Running network')
    tot_loss = []
    coord_loss = []
    conf_loss = []
    cls_loss = []
    anno, det = {}, {}
    num_det = 0

    for idx, (data, box) in enumerate(tqdm(loader, total=len(loader))):
        if arguments.cuda:
            data = data.cuda()
        data = torch.autograd.Variable(data, volatile=True)

        output, loss = net(data, box)

        tot_loss.append(net.loss.loss_tot.data[0]*len(box))
        coord_loss.append(net.loss.loss_coord.data[0]*len(box))
        conf_loss.append(net.loss.loss_conf.data[0]*len(box))
        if net.loss.loss_cls is not None:
            cls_loss.append(net.loss.loss_cls.data[0]*len(box))

        key_val = len(anno)
        anno.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(box)})
        det.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(output)})

    log.debug('Computing statistics')

    pr = bbb.pr(det, anno)
    m_ap = round(bbb.ap(*pr)*100, 2)
    tot = round(sum(tot_loss)/len(anno), 5)
    coord = round(sum(coord_loss)/len(anno), 2)
    conf = round(sum(conf_loss)/len(anno), 2)
    if len(cls_loss) > 0:
        cls = round(sum(cls_loss)/len(anno), 2)
        log.test(f'{net.seen//BATCH} mAP:{m_ap}% Loss:{tot} (Coord:{coord} Conf:{conf} Cls:{cls})')
    else:
        log.test(f'{net.seen//BATCH} mAP:{m_ap}% Loss:{tot} (Coord:{coord} Conf:{conf})')

    if arguments.visdom:
        plot_pr(np.array(pr[0]), np.array(pr[1]), name=f'{net.seen//BATCH}: {m_ap}%')

    if arguments.save_det is not None:
        # Note: These detection boxes are the coordinates for the letterboxed images,
        #       you need ln.data.transform.ReverseLetterbox to have the right ones.
        #       Alternatively, you can save the letterboxed annotations, and use those for statistics later on!
        bbb.generate('det_pickle', det, Path(arguments.save_det).with_suffix('.pkl'))
        #bbb.generate('anno_pickle', det, Path('anno-letterboxed_'+arguments.save_det).with_suffix('.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a lightnet network')
    parser.add_argument('weight', help='Path to weight file', default=None)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-v', '--visdom', action='store_true', help='Visualize training data with visdom')
    parser.add_argument('-s', '--save_det', help='Save detections as a brambox pickle file', default=None)
    args = parser.parse_args()

    # Parse arguments
    if args.cuda:
        if not torch.cuda.is_available():
            log.error('CUDA not available')
            args.cuda = False
        else:
            log.debug('CUDA enabled')

    # Test
    test(args)
