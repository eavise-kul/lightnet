#!/usr/bin/env python
#
#   Train a lightnet network with the engine
#   Copyright EAVISE
#

import os
import argparse
from statistics import mean
import torch
from torch.autograd import Variable
import brambox.boxes as bbb

import lightnet as ln
ln.log.level = ln.Loglvl.VERBOSE
#ln.log.color = False

# Parameters
WORKERS = 4
PIN_MEM = True
VISDOM = {'server': 'http://localhost', 'port': 8080, 'env': 'Lightnet'}

CLASSES = 1
NETWORK_SIZE = [416, 416, 3]
RESIZE_RATE = 10
CONF_THRESH = 0.1
NMS_THRESH = 0.4

BATCH = 64 
BATCH_SUBDIV = 8
MAX_BATCHES = 45000                 # Maximum batches to train for (None -> forever)

LEARNING_RATE = 0.0001              # Initial learning rate
MOMENTUM = 0.9
DECAY = 0.0005
LR_STEPS = (100, 25000, 35000)      # Steps at which the learning rate should be scaled
LR_SCALES = (2, 0.5, 0.1)           # Scales to scale the inital learning rate with

BACKUP = 100                        # Initial backup rate 
BP_STEPS = (500, 5000, 10000)       # Steps at which the backup rate should change
BP_RATES = (500, 1000, 10000)       # New values for the backup rate

TEST = 25                           # Initial test rate (only tested in between epochs)
TS_STEPS = (1000, 5000)             # Steps at which the test rate should change
TS_RATES = (50, 100)                # New values for test rate

assert BATCH % BATCH_SUBDIV == 0, 'Batch subdivision should be a divisor of batch size'
assert len(LR_STEPS) == len(LR_SCALES), 'Learning rate scales should have same number of items as steps'
assert len(BP_STEPS) == len(BP_RATES), 'Backup rates should have same number of items as steps'


class CustomEngine(ln.engine.Engine):
    """ This is a custom engine for this training cycle """
    def __init__(self, arguments, **kwargs):
        ln.log(ln.Loglvl.DEBUG, 'Creating network')
        net = ln.models.YoloVoc(CLASSES, arguments.weight, NETWORK_SIZE, CONF_THRESH, NMS_THRESH)
        if arguments.cuda:
            net.cuda()
        optim = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE/BATCH, momentum=MOMENTUM, dampening=0, weight_decay=DECAY*BATCH)

        ln.log(ln.Loglvl.DEBUG, 'Creating datasets')
        train = ln.models.DarknetData(arguments.train, net)
        if arguments.test is not None:
            test = ln.models.DarknetData(arguments.test, net, train=False, augment=False, class_label_map=arguments.names)
        else:
            test = None

        # Super init
        super(CustomEngine, self).__init__(
            net, optim, train, test, arguments.cuda, arguments.visdom,
            batch_size=BATCH, batch_subdivisions=BATCH_SUBDIV, max_batch=MAX_BATCHES,
            class_label_map=arguments.names, backup_folder=arguments.backup,
            **kwargs
            )

    def update(self):
        """ Update rates according to batch """
        # Test rate
        i = 0
        self.test_rate = TEST
        while self.batch > TS_STEPS[i]:
            self.test_rate = TS_RATES[i]
            i += 1
            if i >= len(TS_STEPS):
                break

        # Backup rate
        i = 0
        self.backup_rate = BACKUP
        while self.batch > BP_STEPS[i]:
            self.backup_rate = BP_RATES[i]
            i += 1
            if i >= len(BP_STEPS):
                break

        # Learning rate
        lr = LEARNING_RATE
        for i in range(len(LR_STEPS)):
            if self.batch >= LR_STEPS[i]:
                lr *= LR_SCALES[i]
            else:
                break
        if lr/self.batch_size != self.learning_rate:
            ln.log(ln.Loglvl.VERBOSE, f'Adjusting learning rate [{lr}]')
            self.learning_rate = lr/self.batch_size

        # Resize
        if self.batch % RESIZE_RATE == 0:
            self.network.change_input_dim()

    def train(self):
        tot_loss = []
        coord_loss = []
        conf_loss = []
        cls_loss = []
        
        self.update()
        self.optimizer.zero_grad()

        loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size = self.mini_batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = WORKERS if self.cuda else 0,
            pin_memory = PIN_MEM if self.cuda else False
            )
        for idx, (data, target) in enumerate(loader):
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

                if self.batch % self.backup_rate == 0:
                    self.network.save_weights(os.path.join(self.backup_folder, f'weights_{self.batch}.pt'))

                if self.sigint or self.batch >= self.max_batch or len(self.trainset) - (idx*self.mini_batch_size) < self.batch_size:
                    return

                self.update()

    def test(self):
        tot_loss = []
        anno, det = {}, {}
        num_det = 0
        at = ln.data.AnnoToTensor(self.network)
        at.max = self.testset.max_anno

        saved_dim = self.network.input_dim
        self.network.input_dim = NETWORK_SIZE

        loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size = self.mini_batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = WORKERS if self.cuda else 0,
            pin_memory = PIN_MEM if self.cuda else False,
            collate_fn = ln.data.bbb_collate,
            )
        for idx, (data, target) in enumerate(loader):
            if self.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            target_tensor = torch.stack([at(a) for a in target])

            output, loss = self.network(data, target_tensor)

            tot_loss.append(loss.data[0]*len(target))
            for i in range(len(target)):
                key = len(anno)
                anno[key] = target[i]
                det[key] = ln.data.bbox_to_brambox(output[i], self.network.input_dim, class_label_map=self.class_label_map)
                num_det += len(det[key])

            if self.sigint:
                return

        if num_det > 1:
            pr = bbb.pr(det, anno, class_label_map=self.class_label_map)
            m_ap = bbb.mean_ap(pr)
            loss = round(sum(tot_loss)/len(anno), 5)

            self.log(f'Loss:{loss} mAP:{round(m_ap*100, 2)}%')
            self.visual(pr=pr)
            self.visual(loss=loss, name='Total loss')
        else:
            ln.log(ln.Loglvl.WARN, f'Not enough detections to perform advanced statistics [{num_det}]')
            self.log(f'Loss:{sum(tot_loss)/len(anno)}')
        self.network.input_dim = saved_dim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a lightnet network')
    parser.add_argument('weight', help='Path to weight file', default=None)
    parser.add_argument('train', help='File containing paths to training images')
    parser.add_argument('test', help='File containing paths to test images', nargs='?')
    parser.add_argument('-b', '--backup', help='Backup folder', default='./backup')
    parser.add_argument('-n', '--names', help='Detection names file', default=None)
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

    if args.names is not None:
        with open(args.names, 'r') as f:
            args.names = f.read().splitlines()

    # Train
    eng = CustomEngine(args)
    eng()
