#!/usr/bin/env python
#
#   Train Yolo on VOC
#   Copyright EAVISE
#

import time
import os
import argparse
from statistics import mean
import torch
from torch.autograd import Variable
from torchvision import transforms as tf
import brambox.boxes as bbb

import lightnet as ln
ln.log.level = ln.Loglvl.VERBOSE
#ln.log.color = False

# Parameters
WORKERS = 4
PIN_MEM = True
VISDOM = {'server': 'http://localhost', 'port': 8080, 'env': 'YoloVOC Train'}
ROOT = 'data'
TRAINFILE = f'{ROOT}/train.pkl'

CLASSES = 20
LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
NETWORK_SIZE = (416, 416)
CONF_THRESH = 0.001
NMS_THRESH = 0.4

BATCH = 64 
BATCH_SUBDIV = 8
MAX_BATCHES = 45000

JITTER = 0.2
FLIP = 0.5
HUE = 0.1
SAT = 1.5
VAL = 1.5

LEARNING_RATE = 0.0001
MOMENTUM = 0.9
DECAY = 0.0005
LR_STEPS = (100, 1000, 25000, 35000)
LR_RATES = (0.0005, 0.001, 0.0001, 0.00001)

BACKUP = 100
BP_STEPS = (1000, 10000, 50000)
BP_RATES = (500, 1000, 5000)

RESIZE = 10
RS_STEPS = ()
RS_RATES = ()

assert BATCH % BATCH_SUBDIV == 0, 'Batch subdivision should be a divisor of batch size'


class CustomDataset(ln.data.BramboxData):
    def __init__(self, anno):
        def identify(img_id):
            return f'{ROOT}/VOCdevkit/{img_id}'

        lb  = ln.data.Letterbox(dataset=self)
        rf  = ln.data.RandomFlip(FLIP)
        rc  = ln.data.RandomCrop(JITTER, True, 0.1)
        hsv = ln.data.HSVShift(HUE, SAT, VAL)
        it  = tf.ToTensor()
        img_tf = tf.Compose([hsv, rc, rf, lb, it])
        anno_tf = tf.Compose([rc, rf, lb])

        super(CustomDataset, self).__init__('anno_pickle', anno, NETWORK_SIZE, LABELS, identify, img_tf, anno_tf, RESIZE*BATCH)


class CustomEngine(ln.engine.Engine):
    """ This is a custom engine for this training cycle """
    batch_size = BATCH
    batch_subdivisions = BATCH_SUBDIV
    max_batch = MAX_BATCHES

    def __init__(self, arguments):
        self.cuda = arguments.cuda
        self.backup_folder = arguments.backup

        ln.log(ln.Loglvl.DEBUG, 'Creating network')
        net = ln.models.Yolo(CLASSES, arguments.weight, CONF_THRESH, NMS_THRESH)
        net.postprocess = tf.Compose([
            net.postprocess,
            ln.data.TensorToBrambox(NETWORK_SIZE, LABELS),
        ])
        if arguments.cuda:
            net.cuda()
        optim = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE/BATCH, momentum=MOMENTUM, dampening=0, weight_decay=DECAY*BATCH)
        super(CustomEngine, self).__init__(net, optim, arguments.visdom)

    def start(self):
        """ Starting values """
        ln.log(ln.Loglvl.DEBUG, 'Creating dataset')
        self.trainloader = ln.data.DataLoader(
            CustomDataset(TRAINFILE),
            batch_size = BATCH // BATCH_SUBDIV,
            shuffle = True,
            drop_last = True,
            num_workers = WORKERS if self.cuda else 0,
            pin_memory = PIN_MEM if self.cuda else False,
            collate_fn = ln.data.list_collate,
            )

        self.add_rate('learning_rate', LR_STEPS, [lr/BATCH for lr in LR_RATES])
        self.add_rate('backup_rate', BP_STEPS, BP_RATES, BACKUP)
        self.add_rate('resize_rate', RS_STEPS, RS_RATES, RESIZE)
        self.update_rates()
        self.trainloader.change_input_dim()

    def update(self):
        """ Update """
        if self.batch % self.backup_rate == 0:
            self.network.save_weights(os.path.join(self.backup_folder, f'weights_{self.batch}.pt'))

        if self.batch % self.resize_rate == 0:
            self.trainloader.change_input_dim()

        self.update_rates()

    def train(self):
        tot_loss = []
        coord_loss = []
        conf_loss = []
        cls_loss = []
        
        self.optimizer.zero_grad()

        for idx, (data, target) in enumerate(self.trainloader):
            if self.cuda:
                data = data.cuda()
            data = Variable(data, requires_grad=True)
            
            loss = self.network(data, target)
            loss.backward()
            
            tot_loss.append(self.network.loss.loss_tot.data[0])
            coord_loss.append(self.network.loss.loss_coord.data[0])
            conf_loss.append(self.network.loss.loss_conf.data[0])
            if self.network.loss.loss_cls is not None:
                cls_loss.append(self.network.loss.loss_cls.data[0])

            if self.network.seen % self.batch_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                tot = round(mean(tot_loss), 5)
                coord = round(mean(coord_loss), 2)
                conf = round(mean(conf_loss), 2)
                self.visual(loss=tot, name='Total loss')
                self.visual(loss=coord, name='Coordinate loss')
                self.visual(loss=conf, name='Confidence loss')
                if len(cls_loss) > 0:
                    cls = round(mean(cls_loss), 2)
                    self.visual(loss=cls, name='Class loss')
                    self.log(f'{self.batch} Loss:{tot} (Coord:{coord} Conf:{conf} Cls:{cls})')
                else:
                    self.log(f'{self.batch} Loss:{tot} (Coord:{coord} Conf:{conf})')
                tot_loss = []
                coord_loss = []
                conf_loss = []
                cls_loss = []

                self.update()

                if self.sigint or self.batch >= self.max_batch or (len(self.trainloader) - idx) <= self.batch_subdivisions:
                    return

    def quit(self):
        if self.batch >= self.max_batch or self.sigint:
            self.network.save_weights(os.path.join(self.backup_folder, f'backup.pt'))
            return True
        else:
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a lightnet network')
    parser.add_argument('weight', help='Path to weight file', default=None)
    parser.add_argument('-b', '--backup', help='Backup folder', default='./backup')
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

    if args.visdom:
        args.visdom = VISDOM
    else:
        args.visdom = None

    if not os.path.isdir(args.backup):
        if not os.path.exists(args.backup):
            ln.log(ln.Loglvl.WARN, 'Backup folder does not exist, creating...')
            os.makedirs(args.backup)
        else:
            ln.log(ln.Loglvl.ERROR, 'Backup path is not a folder', ValueError)

    # Train
    eng = CustomEngine(args)
    b1 = eng.batch
    t1 = time.time()
    eng()
    t2 = time.time()
    b2 = eng.batch

    print(f'\nDuration of {b2-b1} batches: {t2-t1} seconds [{round((t2-t1)/(b2-b1), 3)} sec/batch]')
