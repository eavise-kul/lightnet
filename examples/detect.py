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
import torch.nn as nn

import numpy as np
import lightnet as ln
from lightnet.network.layer import *

ln.log.level = ln.Loglvl.VERBOSE

# Parameters
CLASSES = 1
NETWORK_SIZE = [640, 512]
LABELS = ['person']
CONF_THRESH = .24
NMS_THRESH = .4


# Functions
def create_network():
    """ Create the lightnet network """
    net = ln.models.Yolo(CLASSES, args.weight, anchors={'num':5, 'values':[0.52,1.52, 0.70,2.00, 0.88,2.66, 1.16,3.62, 1.63,5.35]})
    net.postprocess = transforms.Compose([
        ln.data.GetBoundingBoxes(net, CONF_THRESH, NMS_THRESH),
        ln.data.TensorToBrambox(network_size=NETWORK_SIZE, class_label_map=LABELS),
    ])

    if args.cuda:
        net.cuda()
    net.eval()

    return net

def tensor_to_csv(filename, t, fopen='wb'):
    """Flatten a tensor to a 1 dimensional array of floats and write
    them to a text file
    """
    t = t.clone()
    t = t.view(-1)
    if t.is_cuda:
        t = t.cpu()
    x = t.numpy()
    with open(filename, fopen) as f:
        np.savetxt(f, x, fmt="%.8f", delimiter=',')

def detect(net, img_path):
    """ Perform a detection """
    # Load image
    img = cv2.imread(img_path)
    im_h, im_w = img.shape[:2]

    # BGR to RGB !!!
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tf = ln.data.Letterbox.apply(img, dimension=NETWORK_SIZE)
    img_tf = transforms.ToTensor()(img_tf)
    img_tf.unsqueeze_(0)
    if args.cuda:
        img_tf = img_tf.cuda()
    img_tf = Variable(img_tf, volatile=True)
    
    # Run detector
    #tensor_to_csv("network_inputs.csv", img_tf.data.clone())
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
    #for layer_count, mod in enumerate(network.modules_recurse()):
    #    if type(mod) == nn.Conv2d:
    #        tensor_to_csv(f'layer_weights_{layer_count}.csv', mod.bias.data)
    #        tensor_to_csv(f'layer_weights_{layer_count}.csv', mod.weight.data, 'ab')
    #    elif type(mod) == Conv2dBatchLeaky:
    #        tensor_to_csv(f'layer_weights_{layer_count}.csv', mod.layer[1].bias.data)
    #        tensor_to_csv(f'layer_weights_{layer_count}.csv', mod.layer[1].weight.data, 'ab')
    #        tensor_to_csv(f'layer_weights_{layer_count}.csv', mod.layer[1].running_mean, 'ab')
    #        tensor_to_csv(f'layer_weights_{layer_count}.csv', mod.layer[1].running_var, 'ab')
    #        tensor_to_csv(f'layer_weights_{layer_count}.csv', mod.layer[0].weight.data, 'ab')

    # Detection
    if len(args.image) > 0:
        for img_name in args.image:
            print(img_name)
            image, output = detect(network, img_name)

            #for layer_count, mod in enumerate(network.modules_recurse()):
            #    if type(mod) == Conv2dBatchLeaky:
            #        tensor_to_csv(f'layer_outputs_{layer_count}.csv', mod.output.data)
            #    elif type(mod) == nn.Conv2d:
            #        # since Conv2d is not part of pytorch, cannot modify it. But its the last
            #        # layer so we can just capture the network output instead
            #        tensor_to_csv(f'layer_outputs_{layer_count}.csv', network.output.data)
            #    elif type(mod) == Reorg:
            #        tensor_to_csv(f'layer_outputs_reorg.csv', mod.output.data)


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
