#
#   Image and annotations preprocessing for lightnet networks
#   The image transformations work with both Pillow and OpenCV images
#   The annotation transformations work with brambox 2 annotations dataframes
#   Copyright EAVISE
#

import random
import collections
import logging
import math
import numpy as np
from PIL import Image, ImageOps
import torch
from .util import BaseTransform, BaseMultiTransform

log = logging.getLogger(__name__)

try:
    import pandas as pd
    import brambox as bb
except ImportError:
    log.warning('Brambox is not installed and thus all data functionality related to it cannot be used')
    pd = None
    bb = None

try:
    import cv2
except ImportError:
    log.warning('OpenCV is not installed and cannot be used')
    cv2 = None

__all__ = ['Crop', 'Letterbox', 'RandomFlip', 'RandomHSV', 'RandomJitter', 'RandomRotate', 'BramboxToTensor']


#
#   Transform to fit
#
class Crop(BaseMultiTransform):
    """ Rescale and crop images/annotations to the right network dimensions.
    This transform will first rescale to the closest (bigger) dimension possible and then take a crop to the exact dimensions.

    Args:
        dimension (tuple, optional): Default size for the letterboxing, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**
        center (Boolean, optional): Whether to take the crop from the center or randomly.
        intersection_threshold(number, optional): Minimal percentage of the annotation's box area that still needs to be inside the crop; Default **0.001**

    Note:
        If the `intersection_threshold` is a tuple of 2 numbers, then they are to be considered as **(width, height)** threshold values.
        Ohterwise the threshold is to be considered as an area threshold.

    Note:
        Create 1 Crop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, dimension=None, dataset=None, center=True, intersection_threshold=0.001):
        self.dimension = dimension
        self.dataset = dataset
        self.center = center
        if self.dimension is None and self.dataset is None:
            raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')

        self.scale = 1
        self.crop = None

    def _get_crop(self, im_w, im_h, net_w, net_h):
        if net_w / im_w >= net_h / im_h:
            self.scale = net_w / im_w
            ds = int(im_h * self.scale - net_h + 0.5)
            dx = 0
            dy = ds // 2 if self.center else random.randint(0, ds)
        else:
            self.scale = net_h / im_h
            ds = int(im_w * self.scale - net_w + 0.5)
            dx = ds // 2 if self.center else random.randint(0, ds)
            dy = 0

        if ds == 0:
            self.crop = None
        else:
            self.crop = (dx, dy, dx + net_w, dy + net_h)

    def _tf_pil(self, img):
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        else:
            net_w, net_h = self.dimension
        im_w, im_h = img.size
        self._get_crop(im_w, im_h, net_w, net_h)

        # Rescale
        if self.scale != 1:
            bands = img.split()
            bands = [b.resize((int(self.scale * im_w + 0.5), int(self.scale * im_h + 0.5))) for b in bands]
            img = Image.merge(img.mode, bands)
            im_w, im_h = img.size

        # Crop
        if self.crop is not None:
            img = img.crop(self.crop)

        return img

    def _tf_cv(self, img):
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        else:
            net_w, net_h = self.dimension
        im_h, im_w = img.shape[:2]
        self._get_crop(im_w, im_h, net_w, net_h)

        # Rescale
        if self.scale != 1:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

        # Crop
        if self.crop is not None:
            img = img[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2]]

        return img

    def _tf_anno(self, anno):
        anno = anno.copy()

        # Rescale
        if self.scale != 1:
            anno.x_top_left *= self.scale
            anno.y_top_left *= self.scale
            anno.width *= self.scale
            anno.height *= self.scale

        # Filter and Crop
        if self.crop is not None:
            cropped = np.empty((4, len(anno.index)), dtype=np.float64)
            cropped[0] = anno.x_top_left.clip(lower=self.crop[0]).values
            cropped[1] = anno.y_top_left.clip(lower=self.crop[1]).values
            cropped[2] = (anno.x_top_left + anno.width).clip(upper=self.crop[2]).values - cropped[0]
            cropped[3] = (anno.y_top_left + anno.height).clip(upper=self.crop[3]).values - cropped[1]

            if isinstance(self.intersection_threshold, collections.Sequence):
                mask = ((cropped[2] / anno.width.values) >= self.intersection_threshold[0]) & ((cropped[3] / anno.height.values) >= self.intersection_threshold[1])
            else:
                mask = ((cropped[2] * cropped[3]) / (anno.width.values * anno.height.values)) >= self.intersection_threshold
            mask = mask & (cropped[2] > 0) & (cropped[3] > 0)

            anno = anno[mask].copy()
            if len(anno.index) == 0:
                return anno

            if self.crop_anno:
                cropped = cropped[:, mask]
                anno.truncated = (cropped[2] * cropped[3]) / ((anno.width * anno.height) / (1 - anno.truncated)).clip(lower=0)
                anno.x_top_left = cropped[0]
                anno.y_top_left = cropped[1]
                anno.width = cropped[2]
                anno.height = cropped[3]

            anno.x_top_left -= self.crop[0]
            anno.y_top_left -= self.crop[1]

            return anno


class Letterbox(BaseMultiTransform):
    """ Rescale images/annotations and add top/bottom borders to get to the right network dimensions.

    Args:
        dimension (tuple, optional): Default size for the letterboxing, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**

    Note:
        Create 1 Letterbox object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, dimension=None, dataset=None, fill_color=127):
        self.dimension = dimension
        self.dataset = dataset
        self.fill_color = fill_color
        if self.dimension is None and self.dataset is None:
            raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')

        self.pad = None
        self.scale = None

    def _tf_pil(self, img):
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        else:
            net_w, net_h = self.dimension
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
            bands = img.split()
            bands = [b.resize((int(self.scale*im_w), int(self.scale*im_h))) for b in bands]
            img = Image.merge(img.mode, bands)
            im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        img = ImageOps.expand(img, border=self.pad, fill=(self.fill_color,)*channels)
        return img

    def _tf_cv(self, img):
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        else:
            net_w, net_h = self.dimension
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
        channels = img.shape[2] if len(img.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        img = cv2.copyMakeBorder(img, self.pad[1], self.pad[3], self.pad[0], self.pad[2], cv2.BORDER_CONSTANT, value=self.fill_color)
        return img

    def _tf_anno(self, anno):
        anno = anno.copy()

        if self.scale is not None:
            anno.x_top_left *= self.scale
            anno.y_top_left *= self.scale
            anno.width *= self.scale
            anno.height *= self.scale
        if self.pad is not None:
            anno.x_top_left += self.pad[0]
            anno.y_top_left += self.pad[1]

        return anno


#
#   Data augmentation
#
class RandomFlip(BaseMultiTransform):
    """ Randomly flip image.

    Args:
        horizontal (Number [0-1]): Chance of flipping the image horizontally
        vertical (Number [0-1], optional): Chance of flipping the image vertically; Default **0**

    Note:
        Create 1 RandomFlip object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, horizontal, vertical=0):
        self.horizontal = horizontal
        self.vertical = vertical
        self.flip_h = False
        self.flip_v = False
        self.im_w = None
        self.im_h = None

    def _get_flip(self):
        self.flip_h = random.random() < self.horizontal
        self.flip_v = random.random() < self.vertical

    def _tf_pil(self, img):
        self._get_flip()
        self.im_w, self.im_h = img.size

        if self.flip_h and self.flip_v:
            img = img.transpose(Image.ROTATE_180)
        elif self.flip_h:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.flip_v:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img

    def _tf_cv(self, img):
        self._get_flip()
        self.im_h, self.im_w = img.shape[:2]

        if self.flip_h and self.flip_v:
            img = cv2.flip(img, -1)
        elif self.flip_h:
            img = cv2.flip(img, 1)
        elif self.flip_v:
            img = cv2.flip(img, 0)

        return img

    def _tf_anno(self, anno):
        anno = anno.copy()

        if self.flip_h and self.im_w is not None:
            anno.x_top_left = self.im_w - anno.x_top_left - anno.width
        if self.flip_v and self.im_h is not None:
            anno.y_top_left = self.im_h - anno.y_top_left - anno.height

        return anno


class RandomHSV(BaseTransform):
    """ Perform random HSV shift on the RGB data.

    Args:
        hue (Number): Random number between -hue,hue is used to shift the hue
        saturation (Number): Random number between 1,saturation is used to shift the saturation; 50% chance to get 1/dSaturation in stead of dSaturation
        value (Number): Random number between 1,value is used to shift the value; 50% chance to get 1/dValue in stead of dValue

    Warning:
        If you use OpenCV as your image processing library, make sure the image is RGB before using this transform.
        By default OpenCV uses BGR, so you must use `cvtColor`_ function to transform it to RGB.

    .. _cvtColor: https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga397ae87e1288a81d2363b61574eb8cab
    """
    def __init__(self, hue, saturation, value):
        self.hue = hue
        self.saturation = saturation
        self.value = value

    def __call__(self, data):
        dh = random.uniform(-self.hue, self.hue)
        ds = random.uniform(1, self.saturation)
        if random.random() < 0.5:
            ds = 1/ds
        dv = random.uniform(1, self.value)
        if random.random() < 0.5:
            dv = 1/dv

        if data is None:
            return None
        elif isinstance(data, Image.Image):
            return self._tf_pil(data, dh, ds, dv)
        elif isinstance(data, np.ndarray):
            return self._tf_cv(data, dh, ds, dv)
        else:
            log.error(f'HSVShift only works with <PIL images> or <OpenCV images> [{type(data)}]')
            return data     # Pass on data to not destroy pipeline with annos

    @staticmethod
    def _tf_pil(img, dh, ds, dv):
        img = img.convert('HSV')
        channels = list(img.split())

        def wrap_hue(x):
            x += int(dh * 255)
            if x > 255:
                x -= 255
            elif x < 0:
                x += 255
            return x

        channels[0] = channels[0].point(wrap_hue)
        channels[1] = channels[1].point(lambda i: min(255, max(0, int(i*ds))))
        channels[2] = channels[2].point(lambda i: min(255, max(0, int(i*dv))))

        img = Image.merge(img.mode, tuple(channels))
        img = img.convert('RGB')
        return img

    @staticmethod
    def _tf_cv(img, dh, ds, dv):
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        def wrap_hue(x):
            x[x >= 360.0] -= 360.0
            x[x < 0.0] += 360.0
            return x

        img[:, :, 0] = wrap_hue(img[:, :, 0] + (360.0 * dh))
        img[:, :, 1] = np.clip(ds * img[:, :, 1], 0.0, 1.0)
        img[:, :, 2] = np.clip(dv * img[:, :, 2], 0.0, 1.0)

        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = (img * 255).astype(np.uint8)
        return img


class RandomJitter(BaseMultiTransform):
    """ Add random jitter to an image, by randomly cropping (or adding borders) to each side.

    Args:
        jitter (Number [0-1]): Indicates how much of the image we can crop
        crop_anno(Boolean, optional): Whether we crop the annotations inside the image crop; Default **False**
        intersection_threshold(tuple(number) or number, optional): Minimal percentage of the annotation's box area that still needs to be inside the crop; Default **0.001**

    Note:
        If the `intersection_threshold` is a tuple of 2 numbers, then they are to be considered as **(width, height)** threshold values.
        Ohterwise the threshold is to be considered as an area threshold.

    Note:
        Create 1 RandomCrop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, jitter, crop_anno=False, intersection_threshold=0.001, fill_color=127):
        self.jitter = jitter
        self.crop_anno = crop_anno
        self.fill_color = fill_color
        self.intersection_threshold = intersection_threshold
        self.crop = None

    def _get_crop(self, im_w, im_h):
        dw, dh = int(im_w*self.jitter), int(im_h*self.jitter)
        crop_left = random.randint(-dw, dw)
        crop_right = random.randint(-dw, dw)
        crop_top = random.randint(-dh, dh)
        crop_bottom = random.randint(-dh, dh)

        self.crop = (crop_left, crop_top, im_w-crop_right, im_h-crop_bottom)
        return self.crop

    def _tf_pil(self, img):
        im_w, im_h = img.size
        crop = self._get_crop(im_w, im_h)
        crop_w = crop[2] - crop[0]
        crop_h = crop[3] - crop[1]
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1

        img = img.crop((max(0, crop[0]), max(0, crop[1]), min(im_w, crop[2]-1), min(im_h, crop[3]-1)))
        img_crop = Image.new(img.mode, (crop_w, crop_h), color=(self.fill_color,)*channels)
        img_crop.paste(img, (max(0, -crop[0]), max(0, -crop[1])))

        return img_crop

    def _tf_cv(self, img):
        im_h, im_w = img.shape[:2]
        crop = self._get_crop(im_w, im_h)

        crop_w = crop[2] - crop[0]
        crop_h = crop[3] - crop[1]
        img_crop = np.ones((crop_h, crop_w) + img.shape[2:], dtype=img.dtype) * self.fill_color

        src_x1 = max(0, crop[0])
        src_x2 = min(crop[2], im_w)
        src_y1 = max(0, crop[1])
        src_y2 = min(crop[3], im_h)
        dst_x1 = max(0, -crop[0])
        dst_x2 = crop_w - max(0, crop[2]-im_w)
        dst_y1 = max(0, -crop[1])
        dst_y2 = crop_h - max(0, crop[3]-im_h)
        img_crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

        return img_crop

    def _tf_anno(self, anno):
        anno = anno.copy()

        # Filter annotations inside crop
        cropped = np.empty((4, len(anno.index)), dtype=np.float64)
        cropped[0] = anno.x_top_left.clip(lower=self.crop[0]).values
        cropped[1] = anno.y_top_left.clip(lower=self.crop[1]).values
        cropped[2] = (anno.x_top_left + anno.width).clip(upper=self.crop[2]).values - cropped[0]
        cropped[3] = (anno.y_top_left + anno.height).clip(upper=self.crop[3]).values - cropped[1]

        if isinstance(self.intersection_threshold, collections.Sequence):
            mask = ((cropped[2] / anno.width.values) >= self.intersection_threshold[0]) & ((cropped[3] / anno.height.values) >= self.intersection_threshold[1])
        else:
            mask = ((cropped[2] * cropped[3]) / (anno.width.values * anno.height.values)) >= self.intersection_threshold
        mask = mask & (cropped[2] > 0) & (cropped[3] > 0)

        anno = anno[mask].copy()
        if len(anno.index) == 0:
            return anno

        # Crop annotations
        if self.crop_anno:
            cropped = cropped[:, mask]
            anno.truncated = (cropped[2] * cropped[3]) * (1 - anno.truncated) / (anno.width * anno.height)
            anno.x_top_left = cropped[0]
            anno.y_top_left = cropped[1]
            anno.width = cropped[2]
            anno.height = cropped[3]

        anno.x_top_left -= self.crop[0]
        anno.y_top_left -= self.crop[1]

        return anno


class RandomRotate(BaseMultiTransform):
    """ Randomly rotate the image/annotations.
    For the annotations we take the smallest possible rectangle that fits the rotated rectangle.

    Args:
        jitter (Number [0-180]): Random number between -jitter,jitter degrees is used to rotate the image

    Note:
        Create 1 RandomRotate object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, jitter):
        self.jitter = jitter
        self.angle = None
        self.im_w = None
        self.im_h = None

    def _get_rotate(self, im_w, im_h):
        self.im_w = im_w
        self.im_h = im_h
        self.angle = random.randint(-self.jitter, self.jitter)

    def _tf_pil(self, img):
        im_w, im_h = img.size
        self._get_rotate(im_w, im_h)
        return img.rotate(self.angle)

    def _tf_cv(self, img):
        im_h, im_w = img.shape[:2]
        self._get_rotate(im_w, im_h)
        M = cv2.getRotationMatrix2D((im_w/2, im_h/2), self.angle, 1)
        return cv2.warpAffine(img, M, (im_w, im_h))

    def _tf_anno(self, anno):
        anno = anno.copy()

        cx, cy = self.im_w/2, self.im_h/2
        rad = math.radians(-self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Rotate anno
        x1_c = anno.x_top_left - cx
        y1_c = anno.y_top_left - cy
        x2_c = x1_c + anno.width
        y2_c = y1_c + anno.height

        x1_r = (x1_c * cos_a - y1_c * sin_a) + cx
        y1_r = (x1_c * sin_a + y1_c * cos_a) + cy
        x2_r = (x2_c * cos_a - y1_c * sin_a) + cx
        y2_r = (x2_c * sin_a + y1_c * cos_a) + cy
        x3_r = (x2_c * cos_a - y2_c * sin_a) + cx
        y3_r = (x2_c * sin_a + y2_c * cos_a) + cy
        x4_r = (x1_c * cos_a - y2_c * sin_a) + cx
        y4_r = (x1_c * sin_a + y2_c * cos_a) + cy
        rot_x = np.stack([x1_r, x2_r, x3_r, x4_r])
        rot_y = np.stack([y1_r, y2_r, y3_r, y4_r])

        # Max rect box
        anno.x_top_left = rot_x.min(axis=0)
        anno.y_top_left = rot_y.min(axis=0)
        anno.width = rot_x.max(axis=0) - anno.x_top_left
        anno.height = rot_y.max(axis=0) - anno.y_top_left

        return anno


#
#   Util
#
class BramboxToTensor(BaseTransform):
    """ Converts a list of brambox annotation objects to a tensor.

    Args:
        dimension (tuple, optional): Default size of the transformed images, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**
        max_anno (Number, optional): Maximum number of annotations in the list; Default **50**
        class_label_map (list, optional): class label map to convert class names to an index; Default **None**

    Return:
        torch.Tensor: tensor of dimension [max_anno, 5] containing [class_idx,center_x,center_y,width,height] for every detection

    Warning:
        To convert annotations to a torch Tensor, you need to convert the `class_label` to an integer. |br|
        For this purpose, this function will first check if the dataframe has a `class_index` column to use.
        Otherwise, it will convert the strings by mapping them to the index of the `class_label_map` argument.
        If no class_label_map is given, it will then try to convert the class_label to an integer, using `astype(int)`.
        If that fails, it is simply given the number 0.
    """
    def __init__(self, dimension=None, dataset=None, max_anno=50, class_label_map=None):
        self.dimension = dimension
        self.dataset = dataset
        self.max_anno = max_anno
        self.class_label_map = class_label_map

        if self.dimension is None and self.dataset is None:
            raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')
        if self.class_label_map is None:
            log.warning('No class_label_map given. If there is no class_index column or if the class_labels are not integers, they will be set to zero.')

    def __call__(self, data):
        if self.dataset is not None:
            dim = self.dataset.input_dim
        else:
            dim = self.dimension
        return self.apply(data, dim, self.max_anno, self.class_label_map)

    @classmethod
    def apply(cls, data, dimension, max_anno=None, class_label_map=None):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f'BramboxToTensor only works with brambox annotation dataframes [{type(data)}]')

        anno_np = cls._tf_anno(data, dimension, class_label_map)

        if max_anno is not None:
            anno_len = len(anno_np)
            if anno_len > max_anno:
                raise ValueError(f'More annotations than maximum allowed [{anno_len}/{max_anno}]')

            z_np = np.zeros((max_anno-anno_len, 5), dtype=np.float32)
            z_np[:, 0] = -1

            if anno_len > 0:
                return torch.from_numpy(np.concatenate((anno_np, z_np)))
            else:
                return torch.from_numpy(z_np)
        else:
            return torch.from_numpy(anno_np)

    @staticmethod
    def _tf_anno(anno, dimension, class_label_map):
        net_w, net_h = dimension

        if 'class_index' not in anno.columns:
            if class_label_map is not None:
                cls_idx = anno.class_label.map(dict((l, i) for i, l in enumerate(class_label_map))).values
            else:
                try:
                    cls_idx = anno.class_label.astype(int).values
                except ValueError:
                    cls_idx = np.array([0] * len(anno))
        else:
            cls_idx = anno['class_index'].values

        w = anno.width.values / net_w
        h = anno.height.values / net_h
        cx = anno.x_top_left.values / net_w + (w / 2)
        cy = anno.y_top_left.values / net_h + (h / 2)

        return np.stack([cls_idx, cx, cy, w, h], axis=-1).astype(np.float32)
