#!/usr/bin/env python
#
#   Test single image with Lightnet network
#   Copyright EAVISE
#

import time
from statistics import mean
import os
import argparse
import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable
import brambox.boxes as bbb

import lightnet as ln
ln.log.level = ln.Loglvl.VERBOSE

# Parameters
CLASSES = 20
NETWORK_SIZE = [416, 416, 3]
CONF_THRESH = .25
NMS_THRESH = .4

TIMER = False
TIMES_NETWORK = []


# Functions
def create_network():
    """ Create the lightnet network """
    net = ln.models.YoloVoc(CLASSES, args.weight, NETWORK_SIZE, CONF_THRESH, NMS_THRESH)

    if args.cuda:
        net.cuda()
    net.eval()

    return net

def detect(net, img_path):
    """ Perform a detection """
    # Load image
    img = cv2.imread(img_path)
    preprocess = transforms.Compose([
            ln.data.Letterbox(net),
            transforms.ToTensor()
        ])
    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)
    if args.cuda:
        img_tensor = img_tensor.cuda()
    
    # Run detector
    in_var = Variable(img_tensor, volatile=True)
    if TIMER:
        t0 = time.time()
        out = net(in_var)
        t1 = time.time()
        TIMES_NETWORK.append(t1-t0)
    else:
        out = net(in_var)

    return img, out

def draw_boxes(img, det, net_dim):
    """ Draw detections and annotations on the image """
    det = ln.data.bbox_to_brambox(det, net_dim, (img.shape[1], img.shape[0]), names)
    bbb.draw_box(img, det, show_labels=args.label, inline=True)
    return img

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a single image through a lightnet network')
    parser.add_argument('weight', help='Path to weight file')
    parser.add_argument('image', help='Path to image file', nargs='*')
    parser.add_argument('-n', '--names', help='path to names file', default=None)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-s', '--save', action='store_true', help='Save image in stead of displaying it')
    parser.add_argument('-l', '--label', action='store_true', help='Print labels and scores on the image')
    args = parser.parse_args()

    # Parse Arguments
    if args.cuda and not torch.cuda.is_available():
        ln.log(ln.Loglvl.ERROR, 'CUDA not available')
        args.cuda = False

    if args.names is not None:
        with open(args.names, 'r') as f:
            names = f.read().splitlines()
    else:
        names = None

    # Network
    network = create_network()

    # Detection
    if len(args.image) > 0:
        for img_name in args.image:
            print(img_name)
            image, output = detect(network, img_name)
            image = draw_boxes(image, output[0], network.input_dim[:2])
            if args.save:
                cv2.imwrite('detections.png', image)
            else:
                cv2.imshow('image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        while True:
            try:
                img_path = input('Enter image path: ')    
            except (KeyboardInterrupt, EOFError):
                print('')
                break
        
            if not os.path.isfile(img_path):
                print(f'\'{img_path}\' is not a valid path')
                break

            image, output = detect(network, img_path)
            image = draw_boxes(image, output[0], network.input_dim[:2])
            if args.save:
                cv2.imwrite('detections.png', image)
            else:
                cv2.imshow('image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # Timer
    if TIMER:
        print('TIMES NETWORK RUN:')
        print(f'\tMinimum: {min(TIMES_NETWORK)}')
        print(f'\tAverage: {mean(TIMES_NETWORK)}')
        print(f'\tMaximum: {max(TIMES_NETWORK)}')
