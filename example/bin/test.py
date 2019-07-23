#!/usr/bin/env python
import os
import argparse
import logging
from statistics import mean
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import lightnet as ln
import brambox as bb
from dataset import *

log = logging.getLogger('lightnet.VOC.test')


class TestEngine:
    def __init__(self, params, dataloader, **kwargs):
        self.params = params
        self.dataloader = dataloader

        # extract data from params
        self.post = params.post
        self.loss = params.loss
        self.network = params.network

        # Setting kwargs
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
            else:
                log.error('{k} attribute already exists on TestEngine, not overwriting with `{v}`')

    def __call__(self):
        self.params.to(self.device)
        self.network.eval()
        self.loss.eval()    # This is necessary so the loss doesnt use its 'prefill' rule

        if self.loss_format == 'none':
            anno, det = self.test_none()
        else:
            anno, det = self.test_loss()

        aps = []
        for c in tqdm(self.params.class_label_map):
            anno_c = anno[anno.class_label == c]
            det_c = det[det.class_label == c]

            # By default brambox considers ignored annos as regions -> we want to consider them as annos still
            matched_det = bb.stat.match_det(det_c, anno_c, 0.5, criteria=bb.stat.coordinates.iou, ignore=bb.stat.IgnoreMethod.SINGLE)
            pr = bb.stat.pr(matched_det, anno_c)

            aps.append(bb.stat.ap(pr))

        m_ap = round(100 * mean(aps), 2)
        print(f'mAP: {m_ap:.2f}%')

        if self.detection is not None:
            def get_img_dim(name):
                with Image.open(f'data/VOCdevkit/{name}.jpg') as img:
                    return img.size

            rlb = ln.data.transform.ReverseLetterbox(self.params.input_dimension, get_img_dim)
            det = rlb(det)
            bb.io.save(det, 'pandas', self.detection)

    def test_none(self):
        anno, det = [], []

        with torch.no_grad():
            for idx, (data, target) in enumerate(tqdm(self.dataloader)):
                data = data.to(self.device)
                output = self.network(data)
                output = self.post(output)

                output.image = pd.Categorical.from_codes(output.image, dtype=target.image.dtype)
                anno.append(target)
                det.append(output)

        anno = bb.util.concat(anno, ignore_index=True, sort=False)
        det = bb.util.concat(det, ignore_index=True, sort=False)
        return anno, det

    def test_loss(self):
        loss_dict = {'tot': [], 'coord': [], 'conf': [], 'cls': []}
        anno, det = [], []

        with torch.no_grad():
            for idx, (data, target) in enumerate(tqdm(self.dataloader)):
                data = data.to(self.device)
                output = self.network(data)
                loss = self.loss(output, target)
                output = self.post(output)

                num_img = data.shape[0]
                loss_dict['tot'].append(self.loss.loss_tot.item() * num_img)
                loss_dict['coord'].append(self.loss.loss_coord.item() * num_img)
                loss_dict['conf'].append(self.loss.loss_conf.item() * num_img)
                loss_dict['cls'].append(self.loss.loss_cls.item() * num_img)

                output.image = pd.Categorical.from_codes(output.image, dtype=target.image.dtype)
                anno.append(target)
                det.append(output)

        anno = bb.util.concat(anno, ignore_index=True, sort=False)
        det = bb.util.concat(det, ignore_index=True, sort=False)

        loss_tot = sum(loss_dict['tot']) / len(anno.image.cat.categories)
        loss_coord = sum(loss_dict['coord']) / len(anno.image.cat.categories)
        loss_conf = sum(loss_dict['conf']) / len(anno.image.cat.categories)
        loss_cls = sum(loss_dict['cls']) / len(anno.image.cat.categories)
        if self.loss == 'percent':
            loss_coord *= 100 / loss_tot
            loss_conf *= 100 / loss_tot
            loss_cls *= 100 / loss_tot
            log.info(f'Loss:{loss_tot:.5f} (Coord:{loss_coord:.2f}% Conf:{loss_conf:.2f}% Class:{loss_cls:.2f}%)')
        else:
            log.info(f'Loss:{loss_tot:.5f} (Coord:{loss_coord:.2f} Conf:{loss_conf:.2f} Class:{loss_cls:.2f})')

        return anno, det


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test trained network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('weight', help='Path to weight file')
    parser.add_argument('-n', '--network', help='network config file', required=True)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-t', '--thresh', help='Detection Threshold', type=float, default=None)
    parser.add_argument('-l', '--loss', help='How to display loss', choices=['abs', 'percent', 'none'], default='abs')
    parser.add_argument('-a', '--anno', help='annotation folder', default='./data')
    parser.add_argument('-d', '--det', help='Detection pandas file', default=None)
    args = parser.parse_args()

    # Parse arguments
    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            log.debug('CUDA enabled')
            device = torch.device('cuda')
        else:
            log.error('CUDA not available')

    params = ln.engine.HyperParameters.from_file(args.network)
    if args.weight.endswith('.state.pt'):
        params.load(args.weight)
    else:
        params.network.load(args.weight)

    if args.thresh is not None: # Overwrite threshold
        params.post[0].conf_thresh = args.thresh

    # Dataloader
    testing_dataloader = torch.utils.data.DataLoader(
        VOCDataset(os.path.join(args.anno, params.test_set), params, False),
        batch_size = params.mini_batch_size,
        shuffle = False,
        drop_last = False,
        num_workers = 8,
        pin_memory = True,
        collate_fn = ln.data.brambox_collate,
    )

    # Start test
    eng = TestEngine(
        params, testing_dataloader,
        device=device,
        loss_format=args.loss,
        detection=args.det,
    )
    eng()
