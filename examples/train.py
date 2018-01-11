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
from torchvision import transforms as tf
import brambox.boxes as bbb

import lightnet as ln
ln.log.level = ln.Loglvl.VERBOSE
#ln.log.color = False

# Parameters
WORKERS = 4
PIN_MEM = True
VISDOM = {'server': 'http://localhost', 'port': 8080, 'env': 'Lightnet'}

CLASSES = 1
NETWORK_SIZE = [416, 416]
IMG_SIZE = [960, 540]
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
LR_RATES = (0.001, 0.0001, 0.00001) # Scales to scale the inital learning rate with

BACKUP = 100                        # Initial backup rate 
BP_STEPS = (500, 5000, 10000)       # Steps at which the backup rate should change
BP_RATES = (500, 1000, 10000)       # New values for the backup rate

TEST = 25                           # Initial test rate (only tested in between epochs)
TS_STEPS = (1000, 5000)             # Steps at which the test rate should change
TS_RATES = (50, 100)                # New values for test rate

RESIZE = 10
RS_STEPS = ()
RS_RATES = ()

assert BATCH % BATCH_SUBDIV == 0, 'Batch subdivision should be a divisor of batch size'


class CustomEngine(ln.engine.Engine):
    """ This is a custom engine for this training cycle """
    def __init__(self, arguments, **kwargs):
        ln.log(ln.Loglvl.DEBUG, 'Creating network')
        net = ln.models.YoloVoc(CLASSES, arguments.weight, CONF_THRESH, NMS_THRESH)
        net.postprocess = tf.Compose([
            net.postprocess,
            ln.data.TensorToBrambox(NETWORK_SIZE, arguments.names),
        ])
        if arguments.cuda:
            net.cuda()
        optim = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE/BATCH, momentum=MOMENTUM, dampening=0, weight_decay=DECAY*BATCH)

        ln.log(ln.Loglvl.DEBUG, 'Creating datasets')
        train = ln.models.DarknetData(arguments.train, input_dimension=NETWORK_SIZE, class_label_map=arguments.names)
        if arguments.test is not None:
            test = ln.models.DarknetData(arguments.test, False, NETWORK_SIZE, class_label_map=arguments.names)
        else:
            test = None

        # Super init
        super(CustomEngine, self).__init__(
            net, optim, train, test, arguments.cuda, arguments.visdom,
            batch_size=BATCH, batch_subdivisions=BATCH_SUBDIV, max_batch=MAX_BATCHES,
            class_label_map=arguments.names, backup_folder=arguments.backup,
            **kwargs
            )

        # Rates
        self.add_rate('learning_rate', LR_STEPS, [lr/BATCH for lr in LR_RATES])
        self.add_rate('backup_rate', BP_STEPS, BP_RATES, BACKUP)
        self.add_rate('test_rate', TS_STEPS, TS_RATES, TEST)
        self.add_rate('resize_rate', RS_STEPS, RS_RATES, RESIZE)

    def start(self):
        """ Starting values """
        self.update_rates()
        self.trainloader = ln.data.DataLoader(
            self.trainset,
            batch_size = BATCH // BATCH_SUBDIV,
            shuffle = True,
            drop_last = True,
            num_workers = WORKERS if self.cuda else 0,
            pin_memory = PIN_MEM if self.cuda else False,
            collate_fn = ln.data.list_collate,
            )
        self.trainloader.change_input_dim()

    def update(self):
        """ Update """
        self.update_rates()

        # Backup
        if self.batch % self.backup_rate == 0:
            self.network.save_weights(os.path.join(self.backup_folder, f'weights_{self.batch}.pt'))

        # Resize
        if self.batch % RESIZE_RATE == 0:
            self.trainloader.change_input_dim()

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

                if self.sigint or self.batch >= self.max_batch or len(self.trainset) - (idx*self.mini_batch_size) < self.batch_size:
                    return

    def test(self):
        tot_loss = []
        anno, det = {}, {}
        num_det = 0

        loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size = self.mini_batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = WORKERS if self.cuda else 0,
            pin_memory = PIN_MEM if self.cuda else False,
            collate_fn = ln.data.list_collate,
            )
        for idx, (data, target) in enumerate(loader):
            if self.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)

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
        self.visual(pr=pr)
        self.visual(loss=loss, name='Total loss')


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
    import time
    eng = CustomEngine(args)
    b1 = eng.batch
    t1 = time.time()
    eng()
    t2 = time.time()
    b2 = eng.batch

    print(f'\nDuration of {b2-b1} batches: {t2-t1} seconds [{(t2-t1)/(b2-b1)}]')
