#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Train the lightnet yolo network using the lightnet engine
#            This example script uses darknet type annotations
#

import os
import argparse
import logging
from statistics import mean
import visdom
import numpy as np
import torch
from torchvision import transforms as tf
import brambox.boxes as bbb
import lightnet as ln

log = logging.getLogger('lightnet.train')
ln.logger.setLogFile('train.log', filemode='w')             # Enable logging of TRAIN and TEST logs
#ln.logger.setConsoleLevel(logging.DEBUG)                   # Enable debug log messages in terminal
#ln.logger.setConsoleColor(False)                           # Disable colored terminal log messages

# Parameters
WORKERS = 4
PIN_MEM = True
TRAINFILE = '.sandbox/data/files.data'                      # Training dataset files
VALIDFILE = '.sandbox/data/files.data'                      # Validation dataset files
VISDOM_PORT = 8080

CLASSES = 1
LABELS = ['person']
IMG_SIZE = [960, 540]
NETWORK_SIZE = [416, 416]
CONF_THRESH = 0.1
NMS_THRESH = 0.4

BATCH = 64 
MINI_BATCH = 8
MAX_BATCHES = 45000

LEARNING_RATE = 0.0001
MOMENTUM = 0.9
DECAY = 0.0005
LR_STEPS = [100, 25000, 35000]
LR_RATES = [0.001, 0.0001, 0.00001]

BACKUP = 5
BP_STEPS = []
BP_RATES = []

TEST = 10
TS_STEPS = []
TS_RATES = []

RESIZE = 10
RS_STEPS = []
RS_RATES = []


class TrainingEngine(ln.engine.Engine):
    """ This is a custom engine for this training cycle """
    batch_size = BATCH
    mini_batch_size = MINI_BATCH
    max_batches = MAX_BATCHES

    def __init__(self, arguments, **kwargs):
        self.cuda = arguments.cuda
        self.backup_folder = arguments.backup
        self.enable_testing = arguments.test
        self.visdom = arguments.visdom

        log.debug('Creating network')
        net = ln.models.Yolo(CLASSES, arguments.weight, CONF_THRESH, NMS_THRESH)
        net.postprocess.append(ln.data.TensorToBrambox(NETWORK_SIZE, LABELS))
        if self.cuda:
            net.cuda()
        optim = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE/BATCH, momentum=MOMENTUM, dampening=0, weight_decay=DECAY*BATCH)

        log.debug('Creating datasets')
        data = ln.data.DataLoader(
            ln.models.DarknetData(TRAINFILE, input_dimension=NETWORK_SIZE, class_label_map=LABELS),
            batch_size = MINI_BATCH,
            shuffle = True,
            drop_last = True,
            num_workers = WORKERS if self.cuda else 0,
            pin_memory = PIN_MEM if self.cuda else False,
            collate_fn = ln.data.list_collate,
        )
        if self.enable_testing is not None:
            self.testloader = torch.utils.data.DataLoader(
                ln.models.DarknetData(VALIDFILE, False, input_dimension=NETWORK_SIZE, class_label_map=LABELS),
                batch_size = MINI_BATCH,
                shuffle = False,
                drop_last = False,
                num_workers = WORKERS if self.cuda else 0,
                pin_memory = PIN_MEM if self.cuda else False,
                collate_fn = ln.data.list_collate,
            )

        super(TrainingEngine, self).__init__(net, optim, data)

    def start(self):
        """ Starting values """
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

        if self.enable_testing is not None:
            self.plot_test_loss = ln.engine.LinePlotter(
                self.visdom,
                'test_loss',
                name='Total loss',
                opts=dict(
                    title='Testing Loss',
                    xlabel='Batch',
                    ylabel='Loss',
                    showlegend=True
                )
            )
            self.plot_test_pr = ln.engine.LinePlotter(
                self.visdom,
                'test_pr',
                name='latest',
                opts=dict(
                    xlabel='Recall',
                    ylabel='Precision',
                    title='Testing PR',
                    xtickmin=0, xtickmax=1,
                    ytickmin=0, ytickmax=1,
                    showlegend=True
                )
            )
            self.add_rate('test_rate', TS_STEPS, TS_RATES, TEST)

        self.dataloader.change_input_dim()

    def process_batch(self, data):
        data, target = data
        if self.cuda:
            data = data.cuda()
        data = torch.autograd.Variable(data, requires_grad=True)

        loss = self.network(data, target)
        loss.backward()

        self.train_loss['tot'].append(self.network.loss.loss_tot.data[0])
        self.train_loss['coord'].append(self.network.loss.loss_coord.data[0])
        self.train_loss['conf'].append(self.network.loss.loss_conf.data[0])
        if CLASSES > 1:
            self.train_loss['cls'].append(self.network.loss.loss_cls.data[0])
    
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

    def test(self):
        tot_loss = []
        anno, det = {}, {}
        num_det = 0

        for idx, (data, target) in enumerate(self.testloader):
            if self.cuda:
                data = data.cuda()
            data = torch.autograd.Variable(data, volatile=True)

            output, loss = self.network(data, target)

            tot_loss.append(loss.data[0]*len(target))
            key_val = len(anno)
            anno.update({key_val+k: v for k,v in enumerate(target)})
            det.update({key_val+k: v for k,v in enumerate(output)})

            if self.sigint:
                return

        pr = bbb.pr(det, anno)
        m_ap = bbb.ap(*pr)
        loss = round(sum(tot_loss)/len(anno), 5)
        self.log(f'Loss:{loss} mAP:{round(m_ap*100, 2)}%')
        self.plot_test_loss(np.array([loss]), np.array([self.batch]))
        self.plot_test_pr.clear()
        self.plot_test_pr(np.array(pr[0]), np.array(pr[1]), update='replace', name=f'{self.batch} - {round(m_ap*100,2)}%')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a lightnet network')
    parser.add_argument('weight', help='Path to weight file', default=None)
    parser.add_argument('-b', '--backup', help='Backup folder', default='./backup')
    parser.add_argument('-t', '--test', action='store_true', help='Enable testing')
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-v', '--visdom', action='store_true', help='Visualize training data with visdom')
    args = parser.parse_args()

    # Parse arguments
    if args.cuda:
        if not torch.cuda.is_available():
            log.error('CUDA not available')
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
    engine = TrainingEngine(args)
    engine()
