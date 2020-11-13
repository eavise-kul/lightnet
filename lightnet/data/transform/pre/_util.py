#
#   Lightnet utilitary pre-processing operators
#   Copyright EAVISE
#

import logging
import numpy as np
import torch
from ..util import BaseTransform, BaseMultiTransform
from ..._imports import pd

__all__ = ['AnnoTransform', 'BramboxToTensor']
log = logging.getLogger(__name__)


class AnnoTransform(BaseMultiTransform):
    """ MultiTransform that allows to modify annotations.
        Its main use is to be able to transform annotations when creating a single :class:`~lightnet.data.transform.Compose` transformation pipeline for both images and annotations.

        Because the Compose pipeline will only run transformations on the image data, unless it is a multi-transform,
        it is necessary to encapsulate functions for annotaions in a such a class.

        Args:
            fn (callable): Function that is called with the annotation dataframe
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def _tf_pil(self, img):
        return img

    def _tf_cv(self, img):
        return img

    def _tf_torch(self, img):
        return img

    def _tf_anno(self, anno):
        return self.fn(anno)


class BramboxToTensor(BaseTransform):
    """ Converts a list of brambox annotation objects to a tensor.

    .. deprecated:: 2.0.0
        |br| This class is deprectated, because you can use brambox dataframes in the loss functions.

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
        super().__init__()
        log.deprecated('This class is deprecated and will be removed in a future version of lightnet. Please use brambox dataframes instead of tensors in the various loss functions.')
        self.dimension = dimension
        self.dataset = dataset
        self.max_anno = max_anno
        self.class_label_map = class_label_map

        if self.dimension is None and self.dataset is None:
            raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')
        if self.class_label_map is None:
            log.warning('No class_label_map given. If there is no class_index column or if the class_labels are not integers, they will be set to zero.')

    def forward(self, data):
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
