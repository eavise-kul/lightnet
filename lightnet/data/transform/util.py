#
#   Lightnet related data processing
#   Utilitary classes and functions for the data subpackage
#   Copyright EAVISE
#

import logging
from abc import ABC, abstractmethod
import numpy as np
from .._imports import pd, Image

__all__ = ['Compose']
log = logging.getLogger(__name__)


class BaseTransform(ABC):
    """ Base transform class for the pre- and post-processing functions.
    This class allows to create an object with some case specific settings, and then call it with the data to perform the transformation.
    It also allows to call the static method ``apply`` with the data and settings. This is usefull if you want to transform a single data object.
    """
    @abstractmethod
    def __call__(self, data):
        return data

    @classmethod
    def apply(cls, data, **kwargs):
        """ Classmethod that applies the transformation once.

        Args:
            data: Data to transform (eg. image)
            **kwargs: Same arguments that are passed to the ``__init__`` function
        """
        obj = cls(**kwargs)
        return obj(data)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        string = f'{self.__class__.__name__} (\n'

        for name in sorted(self.__dict__.keys()):
            if name.startswith('_'):
                continue
            val = self.__dict__[name]

            valrepr = repr(val)
            if '\n' in valrepr:
                valrepr = val.__class__.__name__

            string += f'  {name} = {valrepr},\n'

        return string + ')'


class BaseMultiTransform(ABC):
    """ Base multiple transform class that is mainly used in pre-processing functions.
    This class exists for transforms that affect both images and annotations.
    It provides a classmethod ``apply``, that will perform the transormation on one (data, target) pair.
    """
    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, torch.Tensor):
            return self._tf_torch(data)
        elif pd is not None and isinstance(data, pd.DataFrame):
            return self._tf_anno(data)
        elif Image is not None and isinstance(data, Image.Image):
            return self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            return self._tf_cv(data)
        else:
            log.error(f'{self.__class__.__name__} only works with <brambox annotation dataframes>, <PIL images> or <OpenCV images> [{type(data)}]')
            return data

    @classmethod
    def apply(cls, data, target=None, **kwargs):
        """ Classmethod that applies the transformation once.

        Args:
            data: Data to transform (eg. image)
            target (optional): ground truth for that data; Default **None**
            **kwargs: Same arguments that are passed to the ``__init__`` function
        """
        obj = cls(**kwargs)
        res_data = obj(data)

        if target is None:
            return res_data

        res_target = obj(target)
        return res_data, res_target

    @abstractmethod
    def _tf_torch(self, img):
        return img

    @abstractmethod
    def _tf_pil(self, img):
        return img

    @abstractmethod
    def _tf_cv(self, img):
        return img

    @abstractmethod
    def _tf_anno(self, anno):
        return anno

    def __str__(self):
        return f'{self.__class__.__name__} [MULTI-TF]'

    def __repr__(self):
        string = f'{self.__class__.__name__} [MULTI-TF] (\n'

        for name in sorted(self.__dict__.keys()):
            if name.startswith('_'):
                continue
            val = self.__dict__[name]

            valrepr = repr(val)
            if '\n' in valrepr:
                valrepr = val.__class__.__name__

            string += f'  {name} = {valrepr},\n'

        return string + ')'


class Compose(list):
    """ This is lightnet's own version of :class:`torchvision.transforms.Compose`, which has some extra bells and whistles.

    One of its main features is that you can create a single pipeline for both your images and annotations.
    When you want to run your pipeline, you simply call this compose list with a tuple,
    containing your image and your annotations (in that specific order).
    This class will then run through the transformations and apply each of them.
    If a transformation is of a type :class:`~lightnet.data.transform.util.BaseMultiTransform`,
    all elements from the data tuple will be run through this transformation sequentially,
    otherwise only the first item will be transformed.

    If you pass anything other than a tuple, it will just be transformed by each transformation sequentially.

    Args:
        transformations (list of callables): A list of all your transformations in the right order.

    Attributes:
        self.multi_tf (tuple): Which classes to consider to be multi-transforms that act on both images and annotations; Default **(BaseMultiTransform,)**

    Example:
        >>> tf = ln.data.transform.Compose([lambda n: n+1])
        >>> tf(10)  # 10+1
        11
        >>> tf.append(lambda n: n*2)
        >>> tf(10)  # (10+1)*2
        22
        >>> tf.insert(0, lambda n: n//2)
        >>> tf(10)  # ((10//2)+1)*2
        12
        >>> del tf[2]
        >>> tf(10)  # (10//2)+1
        6
    """
    multi_tf = (BaseMultiTransform,)

    def __call__(self, data):
        """ Run your data through the transformation pipeline.

        Args:
            data: The data to modify. If it is a tuple, only the first item will be transformed, unless the transform is an instance of self.multi_tf.
        """
        if isinstance(data, tuple):
            for tf in self:
                if isinstance(tf, self.multi_tf):
                    data = tuple(tf(d) for d in data)
                else:
                    data = tuple(tf(d) if i == 0 else d for i, d in enumerate(data))
        else:
            for tf in self:
                data = tf(data)

        return data

    def __getitem__(self, index):
        """ Get a specific item from the transformation list.

        If the index is a string, we compare this string with the classnames of the transformations in the list (all lowercase).
        If there are multiple transforms from the same class, we return the first match.

        If the index is not a string, we simply call the ``__getitem__()`` method from list, which expects an integer.
        """
        if isinstance(index, str):
            index = index.lower()
            keys = tuple(tf.__class__.__name__.lower() for tf in self)
            if index not in keys:
                raise KeyError(f'[{index}] not found in transforms')

            return super().__getitem__(keys.index(index))
        else:
            return super().__getitem__(index)

    def __str__(self):
        string = f'{self.__class__.__name__} ['
        for tf in self:
            string += f'{str(tf)}, '
        return string[:-2] + ']'

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ['
        for tf in self:
            tfrepr = repr(tf)
            if '\n' in tfrepr:
                tfrepr = tfrepr.replace('\n', '\n  ')
            format_string += f'\n  {tfrepr}'
        format_string += '\n]'
        return format_string
