#
#   Image and annotations preprocessing for lightnet networks
#   The image transformations work with both Pillow and OpenCV images
#   The annotation transformations work with brambox.annotations.Annotation objects
#   Copyright EAVISE
#

import random
import collections
import torch
import numpy as np
from PIL import Image, ImageOps
import brambox.boxes as bbb

from ..logger import *

try:
    import cv2
except:
    log(Loglvl.WARN, 'OpenCV is not installed and cannot be used')
    cv2 = None


__all__ = ['Letterbox', 'RandomCrop', 'RandomFlip', 'HSVShift', 'AnnoToTensor']

class Letterbox:
    """ Transform images and annotations to the right network dimensions.

    Args:
        network (lightnet.network.Darknet): Lightnet network that will process the data

    Note:
        Create 1 Letterbox object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, network):
        self.net = network
        self.scale = None
        self.pad = None

    def __call__(self, data):
        if isinstance(data, collections.Sequence):
            return [self._tf_anno(anno) for anno in data]
        elif isinstance(data, Image.Image):
            return self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            return self._tf_cv(data)
        else:
            log(Loglvl.ERROR, f'Letterbox only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]', TypeError)

    def _tf_pil(self, img):
        """ Letterbox an image to fit in the network """
        net_w, net_h = self.net.input_dim[:2]
        im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img

        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            img = img.resize((int(self.scale*im_w), int(self.scale*im_h)))
            im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        img = ImageOps.expand(img, border=self.pad, fill=(127,127,127))
        return img

    def _tf_cv(self, img):
        """ Letterbox and image to fit in the network """
        net_w, net_h = self.net.input_dim[:2]
        im_h, im_w = img.shape[:2]

        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img

        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
            im_h, im_w = img.shape[:2]

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        img = cv2.copyMakeBorder(img, self.pad[1], self.pad[3], self.pad[0], self.pad[2], cv2.BORDER_CONSTANT, value=(127,127,127))
        return img

    def _tf_anno(self, anno):
        """ Change coordinates of an annotation, according to the previous letterboxing """
        if not isinstance(anno, bbb.annotations.Annotation):
            log(Loglvl.ERROR, f'Letterbox only works with lists of <brambox annotations> [{type(anno)}]', TypeError)

        if self.scale is not None:
            anno.rescale(self.scale)
        if self.pad is not None:
            anno.x_top_left += self.pad[0]
            anno.y_top_left += self.pad[1]
        return anno


class RandomCrop:
    """ Take random crop from the image.

    Args:
        jitter (Number [0-1]): Indicates how much of the image we can crop
        crop_anno(Boolean, optional): Whether we crop the annotations inside the image crop; Default **False**
        intersection_threshold(Number [0-1], optional): The minimal percentage an annotation still has to be in the cropped image; Default **0.001**

    Note:
        Create 1 RandomCrop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, jitter, crop_anno=False, intersection_threshold=0.001):
        self.jitter = jitter
        self.crop_anno = crop_anno
        self.inter_thresh = intersection_threshold
        self.crop = None

    def __call__(self, data):
        if isinstance(data, collections.Sequence):
            return list(filter(lambda a:a is not None, [self._tf_anno(anno) for anno in data]))
        elif isinstance(data, Image.Image):
            return self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            return self._tf_cv(data)
        else:
            log(Loglvl.ERROR, f'RandomCrop only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]', TypeError)

    def _tf_pil(self, img):
        """ Take random crop from image """
        im_w, im_h = img.size
        self._get_crop(im_w, im_h)

        return img.crop((self.crop[0], self.crop[1], self.crop[2]-1, self.crop[3]-1))

    def _tf_cv(self, img):
        """ Take random crop from image """
        im_h, im_w = img.shape[:2]
        self._get_crop(im_w, im_h)

        crop_w = self.crop[2] - self.crop[0]
        crop_h = self.crop[3] - self.crop[1]
        img_crop = np.zeros((crop_h, crop_w) + img.shape[2:], dtype=img.dtype)

        src_x1 = max(0, self.crop[0])
        src_x2 = min(self.crop[2], im_w)
        src_y1 = max(0, self.crop[1])
        src_y2 = min(self.crop[3], im_h)
        dst_x1 = max(0, -self.crop[0])
        dst_x2 = crop_w - max(0, self.crop[2]-im_w)
        dst_y1 = max(0, -self.crop[1])
        dst_y2 = crop_h - max(0, self.crop[3]-im_h)
        img_crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

        return img_crop

    def _get_crop(self, im_w, im_h):
        dw, dh = int(im_w*self.jitter), int(im_h*self.jitter)
        crop_left = random.randint(-dw, dw)
        crop_right = random.randint(-dw, dw)
        crop_top = random.randint(-dh, dh)
        crop_bottom = random.randint(-dh, dh)
        self.crop = (crop_left, crop_top, im_w-crop_right, im_h-crop_bottom)

    def _tf_anno(self, anno):
        """ Change coordinates of an annotation, according to the previous crop """
        if not isinstance(anno, bbb.annotations.Annotation):
            log(Loglvl.ERROR, f'RandomCrop only works with lists of <brambox annotations> [{type(anno)}]', TypeError)

        if self.crop is not None:
            # Check intersection
            x1 = max(self.crop[0], anno.x_top_left)
            x2 = min(self.crop[2], anno.x_top_left+anno.width)
            y1 = max(self.crop[1], anno.y_top_left)
            y2 = min(self.crop[3], anno.y_top_left+anno.height)
            w = x2-x1
            h = y2-y1
            r1 = w / anno.width
            r2 = h / anno.height
            if w<=0 or h<=0 or r1 < self.inter_thresh or r2 < self.inter_thresh:
                return None

            # Perform crop
            anno.x_top_left -= self.crop[0]
            anno.y_top_left -= self.crop[1]

            if self.crop_anno:
                if anno.x_top_left < 0:
                    anno.width += anno.x_top_left
                    anno.x_top_left = 0
                if anno.y_top_left < 0:
                    anno.height += anno.y_top_left
                    anno.y_top_left = 0
                
                anno.width = min(self.crop[2]-(anno.x_top_left+self.crop[0]), anno.width)
                anno.height = min(self.crop[3]-(anno.y_top_left+self.crop[1]), anno.height)

        return anno


class RandomFlip:
    """ Randomly flip image.

    Args:
        flip_threshold (Number [0-1]): Chance of flipping the image

    Note:
        Create 1 RandomFlip object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, flip_threshold):
        self.thresh = flip_threshold
        self.flip = False
        self.im_w = None

    def __call__(self, data):
        if isinstance(data, collections.Sequence):
            return [self._tf_anno(anno) for anno in data]
        elif isinstance(data, Image.Image):
            return self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            return self._tf_cv(data)
        else:
            log(Loglvl.ERROR, f'RandomFlip only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]', TypeError)

    def _tf_pil(self, img):
        """ Randomly flip image """
        self._get_flip()
        self.im_w = img.size[0]
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
        
    def _tf_cv(self, img):
        """ Randomly flip image """
        self._get_flip()
        self.im_w = img.shape[1]
        if self.flip:
            img = cv2.flip(img, 1)
        return img

    def _get_flip(self):
        self.flip = random.random() < self.thresh

    def _tf_anno(self, anno):
        """ Change coordinates of an annotation, according to the previous flip """
        if not isinstance(anno, bbb.annotations.Annotation):
            log(Loglvl.ERROR, f'RandomFlip only works with lists of <brambox annotations> [{type(anno)}]', TypeError)

        if self.flip and self.im_w is not None:
            anno.x_top_left = self.im_w - anno.x_top_left - anno.width
        return anno


class HSVShift:
    """ Perform random HSV shift on the RGB data.

    Args:
        hue (Number): Random number between -hue,hue is used to shift the hue
        saturation (Number): Random number between 1,saturation is used to shift the saturation; 50% chance to get 1/dSaturation in stead of dSaturation
        value (Number): Random number between 1,value is used to shift the value; 50% chance to get 1/dValue in stead of dValue
    """
    def __init__(self, hue, saturation, value):
        self.hue = hue
        self.sat = saturation
        self.val = value

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, Image.Image):
            return self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            return self._tf_cv(data)
        else:
            log(Loglvl.ERROR, f'HSVShift only works with <PIL images> or <OpenCV images> [{type(data)}]', TypeError)

    def _tf_pil(self, img):
        """ Random hsv shift """
        self._get_hsv()
        img = img.convert('HSV')
        channels = list(img.split())

        def change_hue(x):
            x += int(self.dh*x)
            while x > 255:
                x -= 255
            while x < 0:
                x += 255
            return x

        channels[0] = channels[0].point(change_hue)
        channels[1] = channels[1].point(lambda i:min(255, max(0, int(i*self.ds))))
        channels[2] = channels[2].point(lambda i:min(255, max(0, int(i*self.dv))))

        img = Image.merge(img.mode, tuple(channels))
        img = img.convert('RGB')
        return img

    def _tf_cv(self, img):
        """ Random hsv shift """
        self._get_hsv()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        def change_hue(x):
            x += int(self.dh*x)
            while x > 255:
                x -= 255
            while x < 0:
                x += 255
            return x
        
        img[:,:,0] = np.vectorize(change_hue)(img[:,:,0])
        img[:,:,1] = np.vectorize(lambda i:min(255, max(0, int(i*self.ds))))(img[:,:,1])
        img[:,:,2] = np.vectorize(lambda i:min(255, max(0, int(i*self.dv))))(img[:,:,2])

        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def _get_hsv(self):
        self.dh = random.uniform(-self.hue, self.hue)
        self.ds = random.uniform(1, self.sat)
        if random.random() < 0.5:
            self.ds = 1/self.ds
        self.dv = random.uniform(1, self.val)
        if random.random() < 0.5:
            self.dv = 1/self.dv


class AnnoToTensor:
    """ Converts a list of brambox annotation objects to a tensor.

    Args:
        network (lightnet.network.Darknet): Network that will process the data
        max_anno (Number, optional): Maximum number of annotations in the list; Default **50**
        class_label_map (list, optional): class label map to convert class names to an index; Default **None**

    Return:
        torch.Tensor: tensor of dimension [max_anno, 5] containing [class_idx,center_x,center_y,width,height] for every detection

    Warning:
        If no class_label_map is given, this function will first try to convert the class_label to an integer. If that fails, it is simply given the number 0.
    """
    def __init__(self, network, max_anno=50, class_label_map=None):
        self.net = network
        self.max = max_anno
        self.class_map = class_label_map
        if class_label_map is None:
            log(Loglvl.WARN, 'No class_label_map given. If the class_labels are not integers, they will be set to zero.')

    def __call__(self, data):
        if isinstance(data, collections.Sequence):
            anno_len = len(data)
            if anno_len > self.max:
                log(Loglvl.ERROR, f'More annotations than maximum allowed [{anno_len}/{self.max}]', ValueError)

            z_np = np.zeros((self.max-anno_len, 5), dtype=np.float32)
            z_np[:,0] = -1

            if anno_len > 0:
                anno_np = np.array([self._tf_anno(anno) for anno in data], dtype=np.float32)
                return torch.from_numpy(np.concatenate((anno_np, z_np)))
            else:
                return torch.from_numpy(z_np)
        else:
            log(Loglvl.ERROR, f'AnnoToTensor only works with <brambox annotation lists> [{type(data)}]', TypeError)

    def _tf_anno(self, anno):
        """ Transforms brambox annotation to list """
        if not isinstance(anno, bbb.annotations.Annotation):
            log(Loglvl.ERROR, f'AnnoToTensor only works with lists of <brambox annotations> [{type(anno)}]', TypeError)

        net_w, net_h = self.net.input_dim[:2]

        if self.class_map is not None:
            cls = self.class_map.index(anno.class_label)
        elif isinstance(anno.class_label, str):
            try:
                cls = int(anno.class_label)
            except:
                cls = 0
        else:
            cls = 0

        cx = (anno.x_top_left + (anno.width / 2)) / net_w
        cy = (anno.y_top_left + (anno.height / 2)) / net_h
        w = anno.width / net_w
        h = anno.height / net_h
        return [cls, cx, cy, w, h]
