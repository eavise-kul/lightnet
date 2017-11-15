#!/usr/bin/env python
#
#   Test single image with Lightnet network
#   Copyright EAVISE
#

import sys
import os
from PIL import Image, ImageOps
from torchvision import transforms
from torch.autograd import Variable

from lightnet.models import *
from lightnet.logger import *
log.level = Loglvl.ALL


# Input arguments
weight_path = sys.argv[1]
img_path = sys.argv[2]


# Functions
def detect(img_path):
    """ Perform a detection """
    # Load network
    ext = os.path.splitext(weight_path)[1]
    if ext == 'weight':                 # Darknet style weightfile
        net = YoloVoc(1, weight_path)
    else:                               # Pytorch pickle file
        net = YoloVoc(1, weight_path, True)
    
    net.postprocess.conf_thresh = 0.5
    net.postprocess.nms_thresh = 0.4
    net.cuda()
    net.eval()
    
    # Load image
    img = Image.open(img_path)
    im_w, im_h = img.size
    net_w, net_h = net.input_dim[:2]
    
    if im_w == net_w and im_h == net_h:
        img_tensor = transforms.ToTensor()(img)
    else:
        if (1 - im_w/net_w) >= (1 - im_h/net_h):
            scale = net_w / im_w
        else:
            scale = net_h / im_h

        pad_w = net_w - int(scale*im_w)) // 2
        pad_h = net_h - int(scale*im_h)) // 2
        assert pad_w >= 0, 'pad_w should be positive'
        assert pad_h >= 0, 'pad_h should be positive'

        preprocess = transforms.Compose([
            transforms.Scale((scale_w, scale_h)),
            transforms.Lambda(lambda img: ImageOps.expand(img, border=(pad_w, pad_h, int(pad_w+.5), int(pad_h+.5)), fill=(127,127,127))),
            transforms.ToTensor()
        ])
        img_tensor = preprocess(img)

    img_tensor.unsqueeze_(0)
    
    # Run detector
    in_var = Variable(img_tensor.cuda(), volatile=True)
    t_network = net(in_var)

    return out


# Main
if __name__ == '__main__':
    for i in range(1):
        out = detect(img_path)
        print(out)
