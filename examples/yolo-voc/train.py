#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Train Yolo on Pascal VOC
#

import os
import argparse
import logging
import time
from statistics import mean
import numpy as np
import visdom
import torch
from torchvision import transforms as tf
import brambox.boxes as bbb
import lightnet as ln

log = logging.getLogger('lightnet.train')
#ln.logger.setLogFile('train.log', filemode='w')            # Enable logging of TRAIN and TEST logs
#ln.logger.setConsoleLevel(logging.DEBUG)                   # Enable debug prints in terminal
#ln.logger.setConsoleColor(False)                           # Disable colored terminal output

# Parameters
WORKERS = 8
PIN_MEM = True
ROOT = 'data'
TRAINFILE = f'{ROOT}/train.pkl'
VISDOM_PORT = 8080

CLASSES = 20
LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
NETWORK_SIZE = (416, 416)
CONF_THRESH = 0.001
NMS_THRESH = 0.4

BATCH = 64 
MINI_BATCH = 8
MAX_BATCHES = 45000

JITTER = 0.2
FLIP = 0.5
HUE = 0.1
SAT = 1.5
VAL = 1.5

LEARNING_RATE = 0.0001
MOMENTUM = 0.9
DECAY = 0.0005
LR_STEPS = [250, 25000, 35000]
LR_RATES = [0.001, 0.0001, 0.00001]

BACKUP = 500
BP_STEPS = [5000, 50000]
BP_RATES = [1000, 5000]

RESIZE = 10
RS_STEPS = []
RS_RATES = []


class VOCDataset(ln.models.BramboxDataset):
    def __init__(self, anno):
        def identify(img_id):
            return f'{ROOT}/VOCdevkit/{img_id}.jpg'

        lb  = ln.data.transform.Letterbox(dataset=self)
        rf  = ln.data.transform.RandomFlip(FLIP)
        rc  = ln.data.transform.RandomCrop(JITTER, True, 0.1)
        hsv = ln.data.transform.HSVShift(HUE, SAT, VAL)
        it  = tf.ToTensor()
        img_tf = ln.data.transform.Compose([hsv, rc, rf, lb, it])
        anno_tf = ln.data.transform.Compose([rc, rf, lb])

        super(VOCDataset, self).__init__('anno_pickle', anno, NETWORK_SIZE, LABELS, identify, img_tf, anno_tf)


class VOCTrainingEngine(ln.engine.Engine):
    """ This is a custom engine for this training cycle """
    batch_size = BATCH
    mini_batch_size = MINI_BATCH
    max_batches = MAX_BATCHES

    def __init__(self, arguments, **kwargs):
        self.cuda = arguments.cuda
        self.backup_folder = arguments.backup
        self.visdom = args.visdom

        log.debug('Creating network')
        net = ln.models.Yolo(CLASSES, arguments.weight, CONF_THRESH, NMS_THRESH)
        net.postprocess.append(ln.data.transform.TensorToBrambox(NETWORK_SIZE, LABELS))
        if self.cuda:
            net.cuda()

        log.debug('Creating optimizer')
        optim = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE/BATCH, momentum=MOMENTUM, dampening=0, weight_decay=DECAY*BATCH)

        log.debug('Creating dataloader')
        data = ln.data.DataLoader(
            VOCDataset(TRAINFILE),
            batch_size = self.mini_batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = WORKERS if self.cuda else 0,
            pin_memory = PIN_MEM if self.cuda else False,
            collate_fn = ln.data.list_collate,
        )

        super(VOCTrainingEngine, self).__init__(net, optim, data, **kwargs)

    def start(self):
        log.debug('Creating additional logging objects')
        if CLASSES > 1:
            legend=['Total loss', 'Coordinate loss', 'Confidence loss', 'Class loss']
        else:
            legend=['Total loss', 'Coordinate loss', 'Confidence loss']
        self.plot_train_loss = ln.engine.LinePlotter(
            self.visdom,
            'train_loss',
            opts=dict(
                title='Training Loss',
                xlabel='Batch',
                ylabel='Loss',
                showlegend=True,
                legend=legend,
            )
        )
        self.train_loss = {'tot': [], 'coord': [], 'conf': [], 'cls': []}
        self.add_rate('learning_rate', LR_STEPS, [lr/BATCH for lr in LR_RATES])
        self.add_rate('backup_rate', BP_STEPS, BP_RATES, BACKUP)
        self.add_rate('resize_rate', RS_STEPS, RS_RATES, RESIZE)
        self.dataloader.change_input_dim()

    def process_batch(self, data):
        data, target = data
        if self.cuda:
            data = data.cuda()
        data = torch.autograd.Variable(data, requires_grad=True)

        loss = self.network(data, target)
        loss.backward()

        if torch.__version__.startswith('0.3'):
            self.train_loss['tot'].append(self.network.loss.loss_tot.data[0])
            self.train_loss['coord'].append(self.network.loss.loss_coord.data[0])
            self.train_loss['conf'].append(self.network.loss.loss_conf.data[0])
            if self.network.loss.loss_cls is not None:
                self.train_loss['cls'].append(self.network.loss.loss_cls.data[0])
        else:
            self.train_loss['tot'].append(self.network.loss.loss_tot.item())
            self.train_loss['coord'].append(self.network.loss.loss_coord.item())
            self.train_loss['conf'].append(self.network.loss.loss_conf.item())
            if self.network.loss.loss_cls is not None:
                self.train_loss['cls'].append(self.network.loss.loss_cls.item())
    
    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

        tot = mean(self.train_loss['tot'])
        coord = mean(self.train_loss['coord'])
        conf = mean(self.train_loss['conf'])
        if CLASSES > 1:
            cls = mean(self.train_loss['cls'])
        self.train_loss = {'tot': [], 'coord': [], 'conf': [], 'cls': []}

        if CLASSES > 1:
            self.plot_train_loss(np.array([[tot, coord, conf, cls]]), np.array([self.batch]))
            self.log(f'{self.batch} Loss:{round(tot, 5)} (Coord:{round(coord, 2)} Conf:{round(conf, 2)} Cls:{round(cls, 2)})')
        else:
            self.plot_train_loss(np.array([[tot, coord, conf]]), np.array([self.batch]))
            self.log(f'{self.batch} Loss:{round(tot, 5)} (Coord:{round(coord, 2)} Conf:{round(conf, 2)})')

        if self.batch % self.backup_rate == 0:
            self.network.save_weights(os.path.join(self.backup_folder, f'weights_{self.batch}.pt'))

        if self.batch % self.resize_rate == 0:
            self.dataloader.change_input_dim()

    def quit(self):
        if self.sigint:
            self.network.save_weights(os.path.join(self.backup_folder, f'backup.pt'))
            return True
        elif self.batch >= self.max_batches:
            self.network.save_weights(os.path.join(self.backup_folder, f'final.pt'))
            return True
        else:
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Lightnet-Yolo for the Pascal VOC dataset')
    parser.add_argument('weight', help='Path to initial weight file', default=None)
    parser.add_argument('-b', '--backup', help='Backup folder', default='./backup')
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda to speed up training')
    parser.add_argument('-v', '--visdom', action='store_true', help='Visualize training data with visdom')
    args = parser.parse_args()

    # Parse arguments
    if args.cuda:
        if not torch.cuda.is_available():
            log.debug('CUDA not available')
            args.cuda = False
        else:
            log.debug('CUDA enabled')

    if args.visdom:
        args.visdom = visdom.Visdom(port=VISDOM_PORT)
    else:
        args.visdom = None

    if not os.path.isdir(args.backup):
        if not os.path.exists(args.backup):
            log.warn('Backup folder does not exist, creating...')
            os.makedirs(args.backup)
        else:
            raise ValueError('Backup path is not a folder')

    # Train
    eng = VOCTrainingEngine(args)
    b1 = eng.batch
    t1 = time.time()
    eng()
    t2 = time.time()
    b2 = eng.batch

    print(f'\nDuration of {b2-b1} batches: {t2-t1} seconds [{round((t2-t1)/(b2-b1), 3)} sec/batch]')
