#!/usr/bin/env python
#
#   Evaluate network on a test dataset and write the resulting bounding
#   boxes to (a) file(s). No need for annotation data
#   Copyright EAVISE
#
import os
import torch
import argparse
from tqdm import tqdm
import lightnet as ln
import brambox.boxes as bbb
from torch.utils.data.dataset import Dataset
import torchvision.transforms as tf
from PIL import Image


class RgbImageData(Dataset):
    def __init__(self, image_expression, network_size, stride=1, offset=0):
        super(RgbImageData, self).__init__()

        # create an image transform pipeline
        letterbox  = ln.data.Letterbox(network_size)
        image_to_tensor  = tf.ToTensor()
        self.img_transform = tf.Compose([letterbox, image_to_tensor])

        # image files
        self.image_files = list(bbb.expand(image_expression, stride, offset))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        if not os.path.exists(image_file):
            log(Loglvl.ERROR, f'image file {image_file} does not exists', ValueError)

        img = Image.open(image_file)

        # preprocess image
        img = self.img_transform(img)
        return image_file, img

    def __len__(self):
        return len(self.image_files)


class EvaluateEngine:
    workers = 8
    pin_mem = True

    mini_batch_size = 8
    confidence_thresh = 0.005
    nms_thresh = 0.45
    image_size = [640, 512]
    network_size = image_size
    anchors = {'num':5, 'values':[0.52,1.52, 0.70,2.00, 0.88,2.66, 1.16,3.62, 1.63,5.35]}

    def __init__(self, weight_file, image_expression, use_cuda):
        self.cuda = use_cuda

        ln.log(ln.Loglvl.DEBUG, 'Creating network')
        net = ln.models.Yolo(1, weight_file, self.confidence_thresh, self.nms_thresh, anchors=self.anchors)
        net.postprocess = tf.Compose([
            net.postprocess,
            ln.data.TensorToBrambox(self.network_size, ['person']),
        ])

        if self.cuda:
            net.cuda()
        net.eval()
        self.network = net

        ln.log(ln.Loglvl.DEBUG, 'Creating dataset')
        self.validation_dataloader = torch.utils.data.DataLoader(
            RgbImageData(image_expression, self.network_size),
            batch_size = self.mini_batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = self.workers if self.cuda else 0,
            pin_memory = self.pin_mem if self.cuda else False,
            collate_fn = ln.data.list_collate,
        )

    def __call__(self, fmt, det_out_file):

        dets = {}
        # process all images
        ln.log(ln.Loglvl.DEBUG, 'Starting evaluation')
        for file_names, img_batch in tqdm(self.validation_dataloader, total=len(self.validation_dataloader)):

            # data is a mini batch of preprocessed images
            if self.cuda:
                img_batch = img_batch.cuda()
            img_batch = torch.autograd.Variable(img_batch, volatile=True)

            output = self.network(img_batch)
            dets.update({file_names[k]: v for k,v in enumerate(output)})

        # reverse letterbox all detections so they have pixel coordinates with respect to the original image sizes
        ln.log(ln.Loglvl.DEBUG, 'Reverse letterbox detections')
        bbb.modify(dets, [ln.data.ReverseLetterbox(self.network_size, self.image_size)])

        # save detections to an output format
        ln.log(ln.Loglvl.DEBUG, 'Writing output')
        bbb.generate(fmt, dets, det_out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained network')
    parser.add_argument('input', help='Image set expression')
    parser.add_argument('weights', help='Path to weight file')
    parser.add_argument('output', help='Path for the detection file or folder')
    parser.add_argument('-f', '--format', help='detection format', choices=bbb.detection_formats.keys(), default='det_yaml')
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda', default=True)
    parser.add_argument('-l', '--loglvl', type=int, help='Logging level (0-3)', default=1)
    args = parser.parse_args()

    # Parse arguments
    ln.log.level = args.loglvl

    if args.cuda:
        if not torch.cuda.is_available():
            ln.log(ln.Loglvl.ERROR, 'CUDA not available')
            args.cuda = False
        else:
            ln.log(ln.Loglvl.DEBUG, 'CUDA enabled')

    # Start validation
    eng = EvaluateEngine(args.weights, args.input, args.cuda)
    eng(args.format, args.output)

