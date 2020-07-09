#
#   Lightnet optional dependencies
#   Copyright EAVISE
#
import logging

__all__ = ['pd', 'bb', 'cv2', 'Image', 'ImageOps']
log = logging.getLogger(__name__)

try:
    import pandas as pd
    import brambox as bb
except ModuleNotFoundError:
    log.warning('Brambox is not installed and thus all data functionality related to it cannot be used')
    pd = None
    bb = None

try:
    import cv2
except ModuleNotFoundError:
    log.warning('OpenCV is not installed and cannot be used')
    cv2 = None

try:
    from PIL import Image, ImageOps
except ModuleNotFoundError:
    log.warning('Pillow is not installed and cannot be used')
    Image, ImageOps = None, None
