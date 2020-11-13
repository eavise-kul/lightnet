#
#   Lightnet data augmentation
#   Copyright EAVISE
#

import random
import math
import numpy as np
import torch
from ..util import BaseTransform, BaseMultiTransform
from ..._imports import cv2, Image

__all__ = ['RandomFlip', 'RandomHSV', 'RandomJitter', 'RandomRotate']


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
        super().__init__()
        self.horizontal = horizontal
        self.vertical = vertical
        self.flip_h = False
        self.flip_v = False
        self.im_w = None
        self.im_h = None

    def _get_params(self):
        self.flip_h = random.random() < self.horizontal
        self.flip_v = random.random() < self.vertical

    def _tf_pil(self, img):
        self._get_params()
        self.im_w, self.im_h = img.size

        if self.flip_h and self.flip_v:
            img = img.transpose(Image.ROTATE_180)
        elif self.flip_h:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.flip_v:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img

    def _tf_cv(self, img):
        self._get_params()
        self.im_h, self.im_w = img.shape[:2]

        if self.flip_h and self.flip_v:
            img = cv2.flip(img, -1)
        elif self.flip_h:
            img = cv2.flip(img, 1)
        elif self.flip_v:
            img = cv2.flip(img, 0)

        return img

    def _tf_torch(self, img):
        self._get_params()

        if self.flip_h and self.flip_v:
            img = torch.flip(img, (1, 2))
        elif self.flip_h:
            img = torch.flip(img, (2,))
        elif self.flip_v:
            img = torch.flip(img, (1,))

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

    .. _cvtColor: https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    """
    def __init__(self, hue, saturation, value):
        super().__init__()
        self.hue = hue
        self.saturation = saturation
        self.value = value

    def _get_params(self):
        self.dh = random.uniform(-self.hue, self.hue)

        self.ds = random.uniform(1, self.saturation)
        if random.random() < 0.5:
            self.ds = 1 / self.ds

        self.dv = random.uniform(1, self.value)
        if random.random() < 0.5:
            self.dv = 1 / self.dv

    def _tf_pil(self, img):
        self._get_params()
        img = img.convert('HSV')
        channels = list(img.split())

        def wrap_hue(x):
            x += int(self.dh * 255)
            if x > 255:
                x -= 255
            elif x < 0:
                x += 255
            return x

        channels[0] = channels[0].point(wrap_hue)
        channels[1] = channels[1].point(lambda i: min(255, max(0, int(i*self.ds))))
        channels[2] = channels[2].point(lambda i: min(255, max(0, int(i*self.dv))))

        img = Image.merge(img.mode, tuple(channels))
        img = img.convert('RGB')
        return img

    def _tf_cv(self, img):
        self._get_params()
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        img[:, :, 0] = self.wrap_hue(img[:, :, 0] + (360.0 * self.dh))
        img[:, :, 1] = np.clip(self.ds * img[:, :, 1], 0.0, 1.0)
        img[:, :, 2] = np.clip(self.dv * img[:, :, 2], 0.0, 1.0)

        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = (img * 255).astype(np.uint8)
        return img

    def _tf_torch(self, img):
        self._get_params()

        # Transform to HSV
        maxval, _ = img.max(0)
        minval, _ = img.min(0)
        diff = maxval - minval

        h = torch.zeros_like(diff)
        mask = (diff != 0) & (maxval == img[0])
        h[mask] = (60 * (img[1, mask] - img[2, mask]) / diff[mask] + 360)
        mask = (diff != 0) & (maxval == img[1])
        h[mask] = (60 * (img[2, mask] - img[0, mask]) / diff[mask] + 120)
        mask = (diff != 0) & (maxval == img[2])
        h[mask] = (60 * (img[0, mask] - img[1, mask]) / diff[mask] + 240)
        h %= 360

        s = torch.zeros_like(diff)
        mask = maxval != 0
        s[mask] = diff[mask] / maxval[mask]

        # Random Shift
        h = self.wrap_hue(h + (360 * self.dh))
        s = torch.clamp(self.ds * s, 0, 1)
        v = torch.clamp(self.dv * maxval, 0, 1)

        # Transform to RGB
        c = v * s
        m = v - c
        x = c * (1 - (((h / 60) % 2) - 1).abs())
        cm = c + m
        xm = x + m

        img = torch.stack((m, m, m))
        mask = (h >= 0) & (h <= 60)
        img[0, mask] = cm[mask]
        img[1, mask] = xm[mask]
        mask = (h > 60) & (h <= 120)
        img[0, mask] = xm[mask]
        img[1, mask] = cm[mask]
        mask = (h > 120) & (h <= 180)
        img[1, mask] = cm[mask]
        img[2, mask] = xm[mask]
        mask = (h > 180) & (h <= 240)
        img[1, mask] = xm[mask]
        img[2, mask] = cm[mask]
        mask = (h > 240) & (h <= 300)
        img[0, mask] = xm[mask]
        img[2, mask] = cm[mask]
        mask = (h > 300) & (h <= 360)
        img[0, mask] = cm[mask]
        img[2, mask] = xm[mask]

        return img

    @staticmethod
    def wrap_hue(h):
        h[h >= 360.0] -= 360.0
        h[h < 0.0] += 360.0
        return h


class RandomJitter(BaseMultiTransform):
    """ Add random jitter to an image, by randomly cropping (or adding borders) to each side.

    Args:
        jitter (Number [0-1]): Indicates how much of the image we can crop
        fill_color (int or float, optional): Fill color to be used for padding (if int, will be divided by 255); Default **0.5**

    Note:
        Create 1 RandomCrop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.

    Warning:
        This transformation only modifies the annotations to fit the new origin point of the image.
        It does not crop the annotations to fit inside the new boundaries, nor does it filter annotations that fall outside of these new boundaries.
        Check out :class:`~lightnet.data.transform.FitAnno` for a transformation that does this.
    """
    def __init__(self, jitter, fill_color=0.5):
        super().__init__()
        self.jitter = jitter
        self.fill_color = fill_color if isinstance(fill_color, float) else fill_color / 255
        self.crop = None

    def _get_params(self, im_w, im_h):
        dw, dh = int(im_w*self.jitter), int(im_h*self.jitter)
        crop_left = random.randint(-dw, dw)
        crop_right = random.randint(-dw, dw)
        crop_top = random.randint(-dh, dh)
        crop_bottom = random.randint(-dh, dh)

        self.crop = (crop_left, crop_top, im_w-crop_right, im_h-crop_bottom)

    def _tf_pil(self, img):
        im_w, im_h = img.size
        self._get_params(im_w, im_h)
        crop_w = self.crop[2] - self.crop[0]
        crop_h = self.crop[3] - self.crop[1]
        shape = np.array(img).shape
        channels = shape[2] if len(shape) > 2 else 1

        img = img.crop((max(0, self.crop[0]), max(0, self.crop[1]), min(im_w, self.crop[2]), min(im_h, self.crop[3])))
        img_crop = Image.new(img.mode, (crop_w, crop_h), color=(int(self.fill_color*255),)*channels)
        img_crop.paste(img, (max(0, -self.crop[0]), max(0, -self.crop[1])))

        return img_crop

    def _tf_cv(self, img):
        im_h, im_w = img.shape[:2]
        self._get_params(im_w, im_h)

        crop_w = self.crop[2] - self.crop[0]
        crop_h = self.crop[3] - self.crop[1]
        img_crop = np.ones((crop_h, crop_w) + img.shape[2:], dtype=img.dtype) * int(self.fill_color*255)

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

    def _tf_torch(self, img):
        im_h, im_w = img.shape[-2:]
        self._get_params(im_w, im_h)

        crop_w = self.crop[2] - self.crop[0]
        crop_h = self.crop[3] - self.crop[1]
        img_crop = torch.full((img.shape[0], crop_h, crop_w), self.fill_color, dtype=img.dtype)

        src_x1 = max(0, self.crop[0])
        src_x2 = min(self.crop[2], im_w)
        src_y1 = max(0, self.crop[1])
        src_y2 = min(self.crop[3], im_h)
        dst_x1 = max(0, -self.crop[0])
        dst_x2 = crop_w - max(0, self.crop[2]-im_w)
        dst_y1 = max(0, -self.crop[1])
        dst_y2 = crop_h - max(0, self.crop[3]-im_h)
        img_crop[:, dst_y1:dst_y2, dst_x1:dst_x2] = img[:, src_y1:src_y2, src_x1:src_x2]

        return img_crop

    def _tf_anno(self, anno):
        anno = anno.copy()

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
        super().__init__()
        self.jitter = jitter
        self.angle = None
        self.im_w = None
        self.im_h = None

    def _get_params(self, im_w, im_h):
        self.im_w = im_w
        self.im_h = im_h
        self.angle = random.randint(-self.jitter, self.jitter)

    def _tf_pil(self, img):
        im_w, im_h = img.size
        self._get_params(im_w, im_h)
        return img.rotate(self.angle)

    def _tf_cv(self, img):
        im_h, im_w = img.shape[:2]
        self._get_params(im_w, im_h)
        M = cv2.getRotationMatrix2D((im_w/2, im_h/2), self.angle, 1)
        return cv2.warpAffine(img, M, (im_w, im_h))

    def _tf_torch(self, img):
        raise NotImplementedError('Random Rotate is not implemented for torch Tensors, you can use Kornia [https://github.com/kornia/kornia]')

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
