#!/usr/bin/env python
#
#   Train a lightnet network
#   Copyright EAVISE
#

import os
import sys
import argparse
from statistics import mean
import numpy as np
import torch
from torch.autograd import Variable
import brambox.boxes as bbb

import lightnet as ln
import lightnet.data as lnd
import lightnet.models as lnm
ln.log.level = ln.Loglvl.VERBOSE
#ln.log.color = False

# Parameters
CLASSES = 1
NETWORK_SIZE = [416, 416, 3]
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
TS_RATES = (50, 25)                # New values for test rate


# Parameter checks
assert BATCH % BATCH_SUBDIV == 0, 'Batch subdivision should be a divisor of batch size'
assert len(LR_STEPS) == len(LR_SCALES), 'Learning rate scales should have same number of items as steps'
assert len(BP_STEPS) == len(BP_RATES), 'Backup rates should have same number of items as steps'


def create_network():
    """ Create the lightnet network and optimizer """
    net = lnm.YoloVoc(CLASSES, args.weight, NETWORK_SIZE, CONF_THRESH, NMS_THRESH)
    if args.cuda:
        net.cuda()

    optim = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE/BATCH, momentum=MOMENTUM, dampening=0, weight_decay=DECAY*BATCH)

    return net, optim

def adjust_learning_rate(optim, batch):
    """ Adjust the learning rate according to the batch number """
    lr = LEARNING_RATE
    for i in range(len(LR_STEPS)):
        if batch >= LR_STEPS[i]:
            lr *= LR_SCALES[i]
        else:
            break

    ln.log(ln.Loglvl.VERBOSE, f'Adjusting learning rate [{lr}]')
    for param_group in optim.param_groups:
        param_group['lr'] = lr / BATCH

    if batch < LR_STEPS[i]:
        return LR_STEPS[i]

def adjust_backup_rate(batch):
    """ Adjust backup rate according to the batch number """
    global BACKUP

    if batch < BP_STEPS[0]:
        return BP_STEPS[0]

    idx = 0
    for i in range(len(BP_STEPS)):
        if batch >= BP_STEPS[i]:
            idx = i
        else:
            break

    ln.log(ln.Loglvl.VERBOSE, f'Adjusting backup rate [{BP_RATES[idx]}]')
    BACKUP = BP_RATES[idx]

    if batch < BP_STEPS[i]:
        return BP_STEPS[i]

def adjust_test_rate(batch):
    """ Adjust test rate according to the batch number """
    global TEST

    if batch < TS_STEPS[0]:
        return TS_STEPS[0]

    idx = 0
    for i in range(len(TS_STEPS)):
        if batch >= TS_STEPS[i]:
            idx = i
        else:
            break

    ln.log(ln.Loglvl.VERBOSE, f'Adjusting test rate [{TS_RATES[idx]}]')
    TEST = TS_RATES[idx]

    if batch < TS_STEPS[i]:
        return TS_STEPS[i]

def train(net, optim, dataset):
    """ Train the network for 1 epoch """
    global LR_CUR_STEP
    global BP_CUR_STEP
    global TS_CUR_STEP

    net.train()
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH//BATCH_SUBDIV, shuffle=True, drop_last=True, **kwargs)

    # Adjust rates
    batch = net.seen // BATCH
    if LR_CUR_STEP is not None and batch >= LR_CUR_STEP:
        LR_CUR_STEP = adjust_learning_rate(optim, batch)
    if BP_CUR_STEP is not None and batch >= BP_CUR_STEP:
        BP_CUR_STEP = adjust_backup_rate(batch)
    if TS_CUR_STEP is not None and batch >= TS_CUR_STEP:
        TS_CUR_STEP = adjust_test_rate(batch)

    tot_minibatch = len(dataset) // (BATCH/BATCH_SUBDIV)
    minibatch = 0
    optim.zero_grad()
    batch_loss = {'total': [], 'coord': [], 'conf': [], 'cls':[]}
    for _, (data, target) in enumerate(loader):
        # Forward & Backward
        if args.cuda:
            data = data.cuda()
        data = Variable(data, requires_grad=True)
        loss = net(data, target)
        loss.backward()
        batch_loss['total'].append(net.loss.loss_tot.data[0])
        batch_loss['coord'].append(net.loss.loss_coord.data[0])
        batch_loss['conf'].append(net.loss.loss_conf.data[0])
        if net.loss.loss_cls is not None:
            batch_loss['cls'].append(net.loss.loss_cls.data[0])

        # Metadata
        minibatch += 1

        # Batch processed
        if net.seen % BATCH == 0:
            batch = net.seen // BATCH
            optim.step()
            optim.zero_grad()

            # Visualisation
            if len(batch_loss['cls']) > 0:
                print(f"[TRAIN]    Batch:{batch} Loss:{round(mean(batch_loss['total']), 5)} (coord:{round(mean(batch_loss['coord']), 2)} conf:{round(mean(batch_loss['conf']), 2)} cls:{mean(batch_loss['cls'])})")
            else:
                print(f"[TRAIN]    Batch:{batch} Loss:{round(mean(batch_loss['total']), 5)} (coord:{round(mean(batch_loss['coord']), 2)} conf:{round(mean(batch_loss['conf']), 2)})")
            batch_loss = {'total': [], 'coord': [], 'conf': [], 'cls':[]}

            # Backup
            if batch % BACKUP == 0:
                net.save_weights(os.path.join(args.backup, f'weights_{batch}.pt'))

            # Resize
            if batch % 10 == 0:
                net.change_input_dim()

            # Adjust rates
            if LR_CUR_STEP is not None and batch >= LR_CUR_STEP:
                LR_CUR_STEP = adjust_learning_rate(optim, batch)
            if BP_CUR_STEP is not None and batch >= BP_CUR_STEP:
                BP_CUR_STEP = adjust_backup_rate(batch)
            if TS_CUR_STEP is not None and batch >= TS_CUR_STEP:
                TS_CUR_STEP = adjust_test_rate(batch)

            # Maximum batches reached
            if batch >= MAX_BATCHES:
                return

            # Not enough images left for new batch
            if tot_minibatch - minibatch < BATCH//BATCH_SUBDIV:
                return

def test(net, dataset):
    """ Test the network """
    original_dim = net.input_dim
    net.input_dim = NETWORK_SIZE
    net.eval()
    kwargs = {'num_workers': 4, 'pin_memory': False} if args.cuda else {}
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH//BATCH_SUBDIV, collate_fn=lnd.list_collate, **kwargs)

    at = lnd.AnnoToTensor(net)
    at.max = dataset.max_anno
    tot_loss = []
    anno, det = {}, {}
    num_det = 0
    for _, (data, target) in enumerate(loader):
        # Forward
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        target_tensor = torch.stack([at(a) for a in target])
        output, loss = net(data, target_tensor)

        # Save output & target
        for i in range(len(target)):
            key = len(anno)
            anno[key] = target[i]
            det[key] = lnd.bbox_to_brambox(output[i], net.input_dim, class_label_map=names)
            num_det += len(det[key])

        # Loss
        tot_loss.append(loss.data[0]*len(target))

    # Compute statistics
    if num_det > 1:
        pr = bbb.pr(det, anno, class_label_map=names)
        m_ap = bbb.mean_ap(pr)

        if args.visdom:
            x = [val[1] for key, val in pr.items()]
            y = [val[0] for key, val in pr.items()]
            legend = [f'{key}: {round(bbb.ap(*val)*100, 2)}' for key, val in pr.items()]
            viz.line(X=np.array(x).transpose(), Y=np.array(y).transpose(),
                win='testset',
                opts=dict(
                title='PR-curve testset',
                xlabel='Recall',
                ylabel='Precision',
                legend=legend,
                xtickmin=0,
                xtickmax=1,
                ytickmin=0,
                ytickmax=1,
                ))
    else:
        ln.log(ln.Loglvl.WARN, f'Not enough detections to perform advanced statistics [{num_det}]')

    # Visualisation
    if num_det > 1:
        print(f'[TEST]     Loss:{round(sum(tot_loss)/len(anno), 5)} mAP:{round(m_ap*100, 2)}%')
    else:
        print(f'[TEST]     Loss:{sum(tot_loss)/len(anno)}')

    # Reset network
    net.input_dim = original_dim


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

    # Parse Arguments
    if args.cuda:
        if not torch.cuda.is_available():
            ln.log(ln.Loglvl.ERROR, 'CUDA not available')
            args.cuda = False
        else:
            ln.log(ln.Loglvl.DEBUG, 'CUDA enabled')

    if args.visdom:
        import visdom
        viz = visdom.Visdom(port=8080)
        ln.log(ln.Loglvl.DEBUG, 'Visdom enabled')

    if not os.path.isdir(args.backup):
        if not os.path.exists(args.backup):
            ln.log(ln.Loglvl.WARN, 'Backup folder does not exist, creating...')
            os.makedirs(args.backup)
        else:
            ln.log(ln.Loglvl.ERROR, 'Backup path is not a folder')
            sys.exit(1)

    if args.names is not None:
        with open(args.names, 'r') as f:
            names = f.read().splitlines()
    else:
        names = None;

    # Create network
    ln.log(ln.Loglvl.DEBUG, 'Creating network')
    network, optimizer = create_network()

    # Set rates
    batch = network.seen//BATCH
    LR_CUR_STEP = adjust_learning_rate(optimizer, batch)
    BP_CUR_STEP = adjust_backup_rate(batch)
    TS_CUR_STEP = adjust_test_rate(batch)
    print()

    # Create Datasets
    ln.log(ln.Loglvl.DEBUG, 'Creating datasets')
    trainset = lnm.DarknetData(args.train, network)
    if args.test is not None:
        testset = lnm.DarknetData(args.test, network, train=False, augment=False, class_label_map=names)
    print()

    # Main loop
    train_batch = network.seen//BATCH - (network.seen//BATCH % TEST)
    while True:
        ln.log(ln.Loglvl.DEBUG, 'Starting train epoch')
        train(network, optimizer, trainset)
        batch = network.seen//BATCH

        if args.test is not None and batch - train_batch >= TEST:
            ln.log(ln.Loglvl.DEBUG, 'Starting test')
            train_batch += TEST
            test(network, testset)
            print()
        
        if MAX_BATCHES is not None and batch >= MAX_BATCHES:
            network.save_weights(os.path.join(args.backup, 'weights_final.pt'))
            ln.log(ln.Loglvl.VERBOSE, f'Reached Maximum batch size [{batch}/{MAX_BATCHES}]')
            if args.test is not None:
                test(network, testset, True)
            break
