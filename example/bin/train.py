#!/usr/bin/env python
import os
import logging
import time
import argparse
from math import isinf
from statistics import mean
import torch
import visdom
import numpy as np
import lightnet as ln
from dataset import *

log = logging.getLogger('lightnet.VOC.train')


class TrainEngine(ln.engine.Engine):
    def start(self):
        self.params.to(self.device)
        self.dataloader.change_input_dim()
        self.optimizer.zero_grad()

        self.train_loss = {'tot': [], 'coord': [], 'conf': [], 'cls': []}
        self.plot_train_loss = ln.engine.LinePlotter(self.visdom, 'train_loss', opts=dict(xlabel='Batch', ylabel='Loss', title='Training Loss', showlegend=True, legend=['Total loss', 'Coordinate loss', 'Confidence loss', 'Class loss']))
        self.plot_lr = ln.engine.LinePlotter(self.visdom, 'learning_rate', name='Learning Rate', opts=dict(xlabel='Batch', ylabel='Learning Rate', title='Learning Rate Schedule'))
        self.batch_end(self.plot_rate)(self.plot)

    def process_batch(self, data):
        data, target = data
        data = data.to(self.device)

        out = self.network(data)
        loss = self.loss(out, target) / self.batch_subdivisions
        loss.backward()

        self.train_loss['tot'].append(self.loss.loss_tot.item())
        self.train_loss['coord'].append(self.loss.loss_coord.item())
        self.train_loss['conf'].append(self.loss.loss_conf.item())
        self.train_loss['cls'].append(self.loss.loss_cls.item())

    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step(self.batch, epoch=self.batch)

        # Get values from last batch
        tot = mean(self.train_loss['tot'][-self.batch_subdivisions:])
        coord = mean(self.train_loss['coord'][-self.batch_subdivisions:])
        conf = mean(self.train_loss['conf'][-self.batch_subdivisions:])
        cls = mean(self.train_loss['cls'][-self.batch_subdivisions:])
        self.log(f'{self.batch} Loss:{tot:.5f} (Coord:{coord:.2f} Conf:{conf:.2f} Cls:{cls:.2f})')

        if isinf(tot):
            log.error('Infinite loss')
            self.sigint = True
            return

    def plot(self):
        tot = mean(self.train_loss['tot'])
        coord = mean(self.train_loss['coord'])
        conf = mean(self.train_loss['conf'])
        cls = mean(self.train_loss['cls'])
        self.train_loss = {'tot': [], 'coord': [], 'conf': [], 'cls': []}

        self.plot_train_loss(np.array([[tot, coord, conf, cls]]), np.array([self.batch]))
        self.plot_lr(np.array([self.optimizer.param_groups[0]['lr']]), np.array([self.batch]))

    @ln.engine.Engine.batch_end(5000)
    def backup(self):
        self.params.save(os.path.join(self.backup_folder, f'weights_{self.batch}.state.pt'))
        log.info(f'Saved backup')

    @ln.engine.Engine.batch_end(10)
    def resize(self):
        self.dataloader.change_input_dim()

    def quit(self):
        if self.batch >= self.max_batches:
            self.params.network.save(os.path.join(self.backup_folder, 'final.pt'))
            return True
        elif self.sigint:
            self.params.save(os.path.join(self.backup_folder, 'backup.state.pt'))
            return True
        else:
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('weight', help='Path to weight file', default=None, nargs='?')
    parser.add_argument('-n', '--network', help='network config file', required=True)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-b', '--backup', metavar='folder', help='Backup folder', default='./backup')
    parser.add_argument('-v', '--visdom', action='store_true', help='Visualize training data with visdom')
    parser.add_argument('-e', '--visdom_env', help='Visdom environment to plot to', default='main')
    parser.add_argument('-p', '--visdom_port', help='Port of the visdom server', type=int, default=8080)
    parser.add_argument('-r', '--visdom_rate', help='How often to plot to visdom (batches)', type=int, default=1)
    parser.add_argument('-a', '--anno', help='annotation folder', default='./data')
    args = parser.parse_args()

    # Parse arguments
    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            log.debug('CUDA enabled')
            device = torch.device('cuda')
        else:
            log.error('CUDA not available')

    if not os.path.isdir(args.backup):
        if not os.path.exists(args.backup):
            log.warn('Backup folder does not exist, creating...')
            os.makedirs(args.backup)
        else:
            raise ValueError('Backup path is not a folder')
    
    if args.visdom:
        visdom = visdom.Visdom(port=args.visdom_port, env=args.visdom_env)
    else:
        visdom = None

    params = ln.engine.HyperParameters.from_file(args.network)
    if args.weight is not None:
        if args.weight.endswith('.state.pt'):
            params.load(args.weight)
        else:
            params.network.load(args.weight, strict=False)  # Disable strict mode for loading partial weights

    # Dataloader
    training_loader = ln.data.DataLoader(
        VOCDataset(os.path.join(args.anno, params.train_set), params, True),
        batch_size = params.mini_batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 8,
        pin_memory = True,
        collate_fn = ln.data.brambox_collate,
    )

    # Start training
    eng = TrainEngine(
        params, training_loader,
        device=device, visdom=visdom, plot_rate=args.visdom_rate, backup_folder=args.backup
    )
    b1 = eng.batch
    t1 = time.time()
    eng()
    t2 = time.time()
    b2 = eng.batch
    log.info(f'Training {b2-b1} batches took {t2-t1:.2f} seconds [{(t2-t1)/(b2-b1):.3f} sec/batch]')
