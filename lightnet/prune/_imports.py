#
#   Lightnet optional dependencies
#   Copyright EAVISE
#
import logging

__all__ = ['onnx', 'onnx_numpy_helper']
log = logging.getLogger(__name__)

try:
    import onnx
    from onnx import numpy_helper as onnx_numpy_helper
except ModuleNotFoundError:
    log.warning('onnx is not installed and thus no pruning functionality will work')
    onnx = None
    onnx_numpy_helper = None
