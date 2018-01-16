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
CLASSES = 1
NETWORK_SIZE = [416, 416]
LABELS = ['person']
CONF_THRESH = .25
NMS_THRESH = .4


# Functions
def create_network():
    """ Create the lightnet network """
    net = ln.models.Yolo(CLASSES, args.weight)
    net.postprocess = transforms.Compose([
        ln.data.GetBoundingBoxes(net, CONF_THRESH, NMS_THRESH),
        ln.data.TensorToBrambox(network_size=NETWORK_SIZE, class_label_map=LABELS),
    ])

    if args.cuda:
        net.cuda()
    net.eval()

    return net

def detect(net, img_path):
    """ Perform a detection """
    # Load image
    img = cv2.imread(img_path)
    im_h, im_w = img.shape[:2]

    img_tf = ln.data.Letterbox.apply(img, dimension=NETWORK_SIZE)
    img_tf = transforms.ToTensor()(img_tf)
    img_tf.unsqueeze_(0)
    if args.cuda:
        img_tf = img_tf.cuda()
    img_tf = Variable(img_tf, volatile=True)
    
    # Run detector
    out = net(img_tf)
    out = ln.data.ReverseLetterbox.apply(out, NETWORK_SIZE, (im_w, im_h))

    return img, out

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
            bbb.draw_boxes(image, output[0], show_labels=args.label)
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
            bbb.draw_boxes(image, output[0], show_labels=args.label)
            if args.save:
                cv2.imwrite('detections.png', image)
            else:
                cv2.imshow('image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
