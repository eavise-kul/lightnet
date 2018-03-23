#
#   Lightnet related data processing
#   These classes and functions provide base functionality that can be used by pre- and post-processing functions
#   Copyright EAVISE
#


class BaseTransform:
    """ Base transform class for the pre- and post-processing functions.
    This class allows to create an object with some case specific settings, and then call it with the data to perform the transformation.
    It also allows to call the static method ``apply`` with the data and settings. This is usefull if you want to transform a single data object.
    """
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __call__(self, data):
        return self.apply(data, **self.__dict__)
    
    @classmethod
    def apply(cls, data, **kwargs):
        """ Classmethod that applies the transformation once.
        
        Args:
            data: Data to transform (eg. image)
            **kwargs: Same arguments that are passed to the ``__init__`` function
        """
        return data


class BaseMultiTransform:
    """ Base multiple transform class that is mainly used in pre-processing functions. 
    This class exists for transforms that affect both images and annotations.
    It provides a classmethod ``apply``, that will perform the transormation on one (data, target) pair.
    """

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
