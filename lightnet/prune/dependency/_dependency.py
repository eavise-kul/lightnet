#
#   Dependency list related functions
#   Copyright EAVISE
#
import logging
import tempfile
import torch
from .._imports import onnx
from ._create import *
from ._tree import *
from ._type import *

__all__ = ['get_dependency_map', 'get_onnx_model', 'print_dependency_map']
log = logging.getLogger(__name__)


def get_dependency_map(model, input_dim):
    """ Computes the dependency map of a model.

    Args:
        model (torch.nn.Module): Network model
        input_dim (tuple <int>): Input dimension for the model, usually [batch, channels, height, width]

    Returns:
        dict: dependency map for each prunable convolution
    """
    # Get ONNX graph
    path = tempfile.TemporaryFile(prefix='lightnet-prune', suffix='.onnx')
    get_onnx_model(model, input_dim, path)
    path.seek(0)
    onnx_model = onnx.load_model(path, load_external_data=False)
    path.close()

    # Create dependency tree
    dependencies = dict()
    for el in onnx_model.graph.node:
        if el.op_type == 'ATen' and el.attribute[0].s == b'_convolution':
            try:
                name = [i for i in el.input if i.endswith('.weight')][0][:-7]
                module = model
                for p in name.split('.'):
                    module = getattr(module, p)
                if module.groups != 1:
                    raise NotImplementedError(f'ATen: Grouped Convolution (cannot prune)')

                dep = create_node(el, onnx_model.graph, model)
                dependencies[dep.name] = dep
            except NotImplementedError as err:
                log.info(f'Cannot prune [{name}], unimplemented dependency [{err}]')
            except StopIteration as err:
                log.info(f'Cannot prune [{name}], generates output [{err}]')

    # Remove ignored and match with modules
    for dep in dependencies.values():
        for _, node in traverse_depth_first(dep):
            if node.type is NodeType.IGNORE:
                for p in node.parents:
                    idx = p.children.index(node)
                    p.children[idx:idx+1] = node.children
                for c in node.children:
                    idx = c.parents.index(node)
                    c.parents[idx:idx+1] = node.parents
                del node
            elif node.name is not None:
                path = node.name.split('.')
                module = model
                try:
                    for p in path:
                        module = getattr(module, p)
                    node.module = module
                except AttributeError:
                    log.debug(f'Could not get PyTorch module for [{node.name}]')
                    continue

    return dependencies


def get_onnx_model(model, input_dim, path):
    """ Compute and save the ONNX version of a model. |br|
    This function computes the ONNX version of a model, which is used to compute the dependency map for pruning.

    Args:
        model (torch.nn.Module): Network model
        input_dim (tuple <int>): Input dimension for the model, usually [batch, channels, height, width]
        path (string): Path to save the ONNX model

    Note:
        This function is used internally by lightnet whilst computing the dependency map
        and should generally not be used by users.
        It is only made publicly available in order to help debug pruning issues.
    """
    # Get device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        log.warn('Could not determine device from model, using "cpu".')
        device = torch.device('cpu')

    # Create input tensor
    input_tensor = torch.rand(*input_dim).to(device)

    # Create onnx model
    torch.onnx.export(
        model, input_tensor, path,
        keep_initializers_as_inputs=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN,
        # Necessary because OperatorExportTypes.ONNX_ATEN
        do_constant_folding=False
    )


def print_dependency_map(dependencies, indentation=' '):
    """ Pretty print a dependency map for easier inspection.

    Args:
        dependencies (dict): The computed dependecy map (see :func:`~lightnet.prune.dependency.get_dependency_map`)
        indentation (string, optional): Indentation to add for each level of dependency; Default **' '**
    """
    for name, d in dependencies.items():
        print(name)
        for i, n in traverse_depth_first(d, True):
            print(f' {2 * i * indentation}{n}')
        print()
