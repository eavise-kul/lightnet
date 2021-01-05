#
#   Lightnet preprocessing for fitting data to certain dimensions
#   Copyright EAVISE
#

import random
import collections
import numpy as np
import torch
from ..util import BaseMultiTransform
from ..._imports import cv2, Image, ImageOps

__all__ = ['Crop', 'Letterbox', 'Pad', 'FitAnno']


class Crop(BaseMultiTransform):
    """ Rescale and crop images/annotations to the right network dimensions.
    This transform will first rescale to the closest (bigger) dimension possible and then take a crop to the exact dimensions.

    Args:
        dimension (tuple, optional): Default size for the letterboxing, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**
        center (Boolean, optional): Whether to take the crop from the center or randomly.

    Note:
        Create 1 Crop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.

    Warning:
        This transformation only modifies the annotations to fit the new scale and origin point of the image.
        It does not crop the annotations to fit inside the new boundaries, nor does it filter annotations that fall outside of these new boundaries.
        Check out :class:`~lightnet.data.transform.FitAnno` for a transformation that does this.
    """
    def __init__(self, dimension=None, dataset=None, center=True):
        super().__init__()
        self.dimension = dimension
        self.dataset = dataset
        self.center = center
        if self.dimension is None and self.dataset is None:
            raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')

        self.scale = 1
        self.crop = None

    def _get_params(self, im_w, im_h):
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        elif isinstance(self.dimension, int):
            net_w, net_h = self.dimension, self.dimension
        else:
            net_w, net_h = self.dimension

        if net_w / im_w >= net_h / im_h:
            self.scale = net_w / im_w
            xcrop = 0
            ycrop = int(im_h * self.scale - net_h + 0.5)
        else:
            self.scale = net_h / im_h
            xcrop = int(im_w * self.scale - net_w + 0.5)
            ycrop = 0

        if xcrop == 0 and ycrop == 0:
            self.crop = None
        else:
            dx = xcrop // 2 if self.center else random.randint(0, xcrop)
            dy = ycrop // 2 if self.center else random.randint(0, ycrop)
            self.crop = (dx, dy, dx + net_w, dy + net_h)

    def _tf_pil(self, img):
        im_w, im_h = img.size
        self._get_params(im_w, im_h)

        # Rescale
        if self.scale != 1:
            bands = img.split()
            bands = [b.resize((int(self.scale*im_w+0.5), int(self.scale*im_h+0.5)), resample=Image.BILINEAR) for b in bands]
            img = Image.merge(img.mode, bands)
            im_w, im_h = img.size

        # Crop
        if self.crop is not None:
            img = img.crop(self.crop)

        return img

    def _tf_cv(self, img):
        im_h, im_w = img.shape[:2]
        self._get_params(im_w, im_h)

        # Rescale
        if self.scale != 1:
            img = cv2.resize(img, (int(im_w*self.scale+0.5), int(im_h*self.scale+0.5)), interpolation=cv2.INTER_LINEAR)

        # Crop
        if self.crop is not None:
            img = img[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2]]

        return img

    def _tf_torch(self, img):
        im_h, im_w = img.shape[-2:]
        self._get_params(im_w, im_h)

        # Rescale
        if self.scale != 1:
            if img.ndim == 3:
                img = img[None, ...]
            elif img.ndim == 2:
                img = img[None, None, ...]
            img = torch.nn.functional.interpolate(
                img,
                size=(int(im_h*self.scale+0.5), int(im_w*self.scale+0.5)),
                mode='bilinear',
                align_corners=False,
            ).squeeze().clamp(min=0, max=255)

        # Crop
        if self.crop is not None:
            img = img[..., self.crop[1]:self.crop[3], self.crop[0]:self.crop[2]]

        return img

    def _tf_anno(self, anno):
        anno = anno.copy()

        # Rescale
        if self.scale != 1:
            anno.x_top_left *= self.scale
            anno.y_top_left *= self.scale
            anno.width *= self.scale
            anno.height *= self.scale

        # Crop
        if self.crop is not None:
            anno.x_top_left -= self.crop[0]
            anno.y_top_left -= self.crop[1]

        return anno


class Letterbox(BaseMultiTransform):
    """ Rescale images/annotations and add top/bottom borders to get to the right network dimensions.

    Args:
        dimension (tuple, optional): Default size for the letterboxing, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**
        fill_color (int or float, optional): Fill color to be used for padding (if int, will be divided by 255); Default **0.5**

    Note:
        Create 1 Letterbox object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, dimension=None, dataset=None, fill_color=0.5):
        super().__init__()
        self.dimension = dimension
        self.dataset = dataset
        self.fill_color = fill_color if isinstance(fill_color, float) else fill_color / 255
        if self.dimension is None and self.dataset is None:
            raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')

        self.pad = None
        self.scale = None

    def _get_params(self, im_w, im_h):
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        elif isinstance(self.dimension, int):
            net_w, net_h = self.dimension, self.dimension
        else:
            net_w, net_h = self.dimension

        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h

        pad_w = (net_w - int(im_w*self.scale+0.5)) / 2
        pad_h = (net_h - int(im_h*self.scale+0.5)) / 2
        if pad_w == 0 and pad_h == 0:
            self.pad = None
        else:
            self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))

    def _tf_pil(self, img):
        im_w, im_h = img.size
        self._get_params(im_w, im_h)

        # Rescale
        if self.scale != 1:
            bands = img.split()
            bands = [b.resize((int(self.scale*im_w+0.5), int(self.scale*im_h+0.5)), resample=Image.BILINEAR) for b in bands]
            img = Image.merge(img.mode, bands)

        # Pad
        if self.pad is not None:
            shape = np.array(img).shape
            channels = shape[2] if len(shape) > 2 else 1
            img = ImageOps.expand(img, border=self.pad, fill=(int(self.fill_color*255),)*channels)

        return img

    def _tf_cv(self, img):
        im_h, im_w = img.shape[:2]
        self._get_params(im_w, im_h)

        # Rescale
        if self.scale != 1:
            img = cv2.resize(img, (int(im_w*self.scale+0.5), int(im_h*self.scale+0.5)), interpolation=cv2.INTER_LINEAR)

        # Pad
        if self.pad is not None:
            channels = img.shape[2] if len(img.shape) > 2 else 1
            img = cv2.copyMakeBorder(img, self.pad[1], self.pad[3], self.pad[0], self.pad[2], cv2.BORDER_CONSTANT, value=(int(self.fill_color*255),)*channels)

        return img

    def _tf_torch(self, img):
        im_h, im_w = img.shape[-2:]
        self._get_params(im_w, im_h)

        # Rescale
        if self.scale != 1:
            if img.ndim == 3:
                img = img[None, ...]
            elif img.ndim == 2:
                img = img[None, None, ...]
            img = torch.nn.functional.interpolate(
                img,
                size=(int(im_h*self.scale+0.5), int(im_w*self.scale+0.5)),
                mode='bilinear',
                align_corners=False,
            ).squeeze().clamp(min=0, max=255)

        # Pad
        if self.pad is not None:
            img = torch.nn.functional.pad(img, (self.pad[0], self.pad[2], self.pad[1], self.pad[3]), value=self.fill_color)

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


class Pad(BaseMultiTransform):
    """ Pad images/annotations to a certain dimension.

    Args:
        dimension (int or tuple, optional): Default size for the padding, expressed as a single integer or as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**
        multiple_dim (boolean, optional): Consider given dimensions to be multiples instead of exact values; Default **True**
        fill_color (int or float, optional): Fill color to be used for padding (if int, will be divided by 255); Default **0.5**

    Warning:
        Do note that the ``dimension`` or ``dataset`` argument here uses the given width and height as a multiple instead of a real dimension by default.
        Given a certain value X, the image (and annotations) will be padded, so that the image dimensions are a multiple of X. |br|
        This is different compared to :class:`~lightnet.data.transform.Crop` or :class:`~lightnet.data.transform.Letterbox`.

        You can toggle this behaviour by setting ``multiple_dim=False``, but keep in mind that the given dimensions should always be bigger than the original input image dimensions.

    Note:
        Create 1 Pad object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, dimension=None, dataset=None, multiple_dim=True, fill_color=127):
        super().__init__()
        self.dimension = dimension
        self.dataset = dataset
        self.multiple_dim = multiple_dim
        self.fill_color = fill_color if isinstance(fill_color, float) else fill_color / 255
        if self.dimension is None and self.dataset is None:
            raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')

        self.pad = None
        self.scale = None

    def _get_params(self, im_w, im_h):
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        elif isinstance(self.dimension, int):
            net_w, net_h = self.dimension, self.dimension
        else:
            net_w, net_h = self.dimension

        if self.multiple_dim:
            if im_w % net_w == 0 and im_h % net_h == 0:
                self.pad = None
            else:
                pad_w = ((net_w - (im_w % net_w)) % net_w) / 2
                pad_h = ((net_h - (im_h % net_h)) % net_h) / 2
                self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        else:
            if im_w == net_w and im_h == net_h:
                self.pad = None
            elif im_w <= net_w and im_h <= net_h:
                pad_w = (net_w - im_w) / 2
                pad_h = (net_h - im_h) / 2
                self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
            else:
                raise ValueError(f'Can only pad to bigger dimensions. Image is bigger than network dimensions [({im_h}, {im_w}) -> ({net_h}, {net_w})]')

    def _tf_pil(self, img):
        im_w, im_h = img.size
        self._get_params(im_w, im_h)

        # Pad
        if self.pad is not None:
            shape = np.array(img).shape
            channels = shape[2] if len(shape) > 2 else 1
            img = ImageOps.expand(img, border=self.pad, fill=(int(self.fill_color*255),)*channels)

        return img

    def _tf_cv(self, img):
        im_h, im_w = img.shape[:2]
        self._get_params(im_w, im_h)

        # Pad
        if self.pad is not None:
            channels = img.shape[2] if len(img.shape) > 2 else 1
            img = cv2.copyMakeBorder(img, self.pad[1], self.pad[3], self.pad[0], self.pad[2], cv2.BORDER_CONSTANT, value=(int(self.fill_color*255),)*channels)

        return img

    def _tf_torch(self, img):
        im_h, im_w = img.shape[-2:]
        self._get_params(im_w, im_h)

        # Pad
        if self.pad is not None:
            img = torch.nn.functional.pad(img, (self.pad[0], self.pad[2], self.pad[1], self.pad[3]), value=self.fill_color)

        return img

    def _tf_anno(self, anno):
        anno = anno.copy()

        if self.pad is not None:
            anno.x_top_left += self.pad[0]
            anno.y_top_left += self.pad[1]

        return anno


class FitAnno(BaseMultiTransform):
    """ Crop and filter annotations to fit inside of the image boundaries. |br|
    This transformation also modifies the `truncated` columns of the annotations, by computing how much of the annotation was cut off.

    Args:
        crop (boolean, optional): Whether to actually crop annotations to fit inside the image boundaries; Default **True**
        filter (boolean, optional): Whether to filter the annotations if they are not completely inside of the image boundaries; Default **True**
        filter_type (string, optional): How to filter ('remove' or 'ignore'); Default **remove**
        filter_threshold (number, optional): Minimal percentage of the bounding box area that still needs to be inside the image; Default **0.001**
        remove_empty (boolean, optional): Whether to remove annotations whose bounding box area is zero (independent of filter args); Default **True**

    Note:
        This transformation does not modify the image data, but is still a multi-transform as it needs to read the image to get the dimensions.
        Create 1 FitAnno object and use it for both image and annotation transforms.

    Note:
        If the `filter_threshold` is a tuple of 2 numbers, then they are to be considered as **(width, height)** threshold values.
        Ohterwise the threshold is to be considered as an area threshold.
    """
    def __init__(self, crop=True, filter=True, filter_type='remove', filter_threshold=0.001, remove_empty=True):
        super().__init__()
        self.crop = crop
        self.filter = filter
        self.filter_type = filter_type.lower()
        self.filter_threshold = filter_threshold
        self.remove_empty = True

        self.image_dim = None

    def _get_params(self, im_w, im_h):
        self.image_dim = (im_w, im_h)

    def _tf_pil(self, img):
        im_w, im_h = img.size
        self._get_params(im_w, im_h)
        return img

    def _tf_cv(self, img):
        im_h, im_w = img.shape[:2]
        self._get_params(im_w, im_h)
        return img

    def _tf_torch(self, img):
        im_h, im_w = img.shape[-2:]
        self._get_params(im_w, im_h)
        return img

    def _tf_anno(self, anno):
        anno = anno.copy()

        crop_coords = np.empty((4, len(anno.index)), dtype=np.float64)
        crop_coords[0] = anno.x_top_left.values
        crop_coords[1] = crop_coords[0] + anno.width.values
        crop_coords[2] = anno.y_top_left.values
        crop_coords[3] = crop_coords[2] + anno.height.values
        crop_coords[:2] = crop_coords[:2].clip(0, self.image_dim[0])
        crop_coords[2:] = crop_coords[2:].clip(0, self.image_dim[1])
        crop_width = crop_coords[1] - crop_coords[0]
        crop_height = crop_coords[3] - crop_coords[2]

        # UserWarnings occur when box width or height is zero (divide by zero)
        # Disable theses annoying warnings as we manually handle the nan cases:
        #   - Masks: `nan >= X = False`
        #   - Computes: `np.nan_to_num(nan) = 0`
        with np.errstate(divide='ignore', invalid='ignore'):
            # Filter
            if self.filter:
                if isinstance(self.filter_threshold, collections.Sequence):
                    mask = (
                        ((crop_width / anno.width.values) >= self.filter_threshold[0])
                        & ((crop_height / anno.height.values) >= self.filter_threshold[1])
                    )
                else:
                    mask = ((crop_width * crop_height) / (anno.width.values * anno.height.values)) >= self.filter_threshold

                if self.filter_type == 'ignore':
                    anno.loc[~mask, 'ignore'] = True
                else:
                    anno = anno[mask].copy()
                    if len(anno.index) == 0:
                        return anno

                    crop_coords = crop_coords[:, mask]
                    crop_width = crop_width[mask]
                    crop_height = crop_height[mask]

            # Crop
            if self.crop:
                anno.truncated = np.nan_to_num((1 - ((crop_width * crop_height * (1 - anno.truncated.values)) / (anno.width.values * anno.height.values))).clip(0, 1))
                anno.x_top_left = crop_coords[0]
                anno.y_top_left = crop_coords[2]
                anno.width = crop_width
                anno.height = crop_height

        # Remove empty
        if self.remove_empty:
            anno = anno[(anno.width > 0) & (anno.height > 0)].copy()

        return anno
