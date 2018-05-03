#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Test the lightnet yolo network on a test data set and compute a PR/mAP metric
#            This example script uses darknet type annotations
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
import brambox.boxes as bbb
import lightnet as ln

log = logging.getLogger('lightnet.test')
ln.logger.setLogFile('test.log', filemode='w')              # Enable logging of TRAIN and TEST logs
#ln.logger.setConsoleLevel(logging.NOTSET)                  # Enable debug prints in terminal
#ln.logger.setConsoleColor(False)                           # Disable colored terminal output

# Parameters
WORKERS = 4
PIN_MEM = True
TESTFILE = '.sandbox/data/files.data'                       # Testing dataset files
VISDOM_PORT = 8080

CLASSES = 1
LABELS = ['person']
NETWORK_SIZE = [416, 416]
CONF_THRESH = 0.1
NMS_THRESH = 0.4

BATCH = 64 
MINI_BATCH = 8


def test(arguments):
    log.debug('Creating network')
    net = ln.models.Yolo(CLASSES, arguments.weight, CONF_THRESH, NMS_THRESH)
    net.postprocess.append(ln.data.transform.TensorToBrambox(NETWORK_SIZE, LABELS))
    net.eval()
    if arguments.cuda:
        net.cuda()

    log.debug('Creating dataset')
    loader = torch.utils.data.DataLoader(
        ln.models.DarknetDataset(TESTFILE, augment=False, input_dimension=NETWORK_SIZE, class_label_map=LABELS),
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
        if torch.__version__.startswith('0.3'):
            data = torch.autograd.Variable(data, volatile=True)
            output, loss = net(data, box)
        else:
            with torch.no_grad():
                output, loss = net(data, box)

        if torch.__version__.startswith('0.3'):
            tot_loss.append(net.loss.loss_tot.data[0]*len(box))
            coord_loss.append(net.loss.loss_coord.data[0]*len(box))
            conf_loss.append(net.loss.loss_conf.data[0]*len(box))
            if net.loss.loss_cls is not None:
                cls_loss.append(net.loss.loss_cls.data[0]*len(box))
        else:
            tot_loss.append(net.loss.loss_tot.item()*len(box))
            coord_loss.append(net.loss.loss_coord.item()*len(box))
            conf_loss.append(net.loss.loss_conf.item()*len(box))
            if net.loss.loss_cls is not None:
                cls_loss.append(net.loss.loss_cls.item()*len(box))

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
