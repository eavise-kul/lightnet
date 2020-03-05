#
#   Categorization of graph operations into NodeType's
#   Copyright EAVISE
#
import enum
import onnx

__all__ = ['NodeType', 'get_node_type', 'add_nodetype_operation']


class NodeType(enum.Enum):
    CONV = enum.auto()          # Convolution operation
    BATCHNORM = enum.auto()     # BatchNorm operation
    CONCAT = enum.auto()        # Concat operation on channel dimension
    ELEMW_OP = enum.auto()      # Element-wise operation
    IGNORE = enum.auto()        # Node that does not need to be modified and does not change output dims
    IGNORE_STOP = enum.auto()   # Ignore node and stop dependency chain


def get_node_type(element):
    op_type = element.op_type
    if op_type == 'ATen':
        attr = onnx.helper.printable_attribute(element.attribute[0])
        attr = attr.split('=')[-1].strip()[1:-1]

        for key, value in node_type_map_aten.items():
            if attr in value:
                return key

        raise NotImplementedError(f'ATen: {attr}')
    else:
        for key, value in node_type_map_onnx.items():
            if op_type in value:
                return key

        raise NotImplementedError(element.op_type)


def add_nodetype_operation(name, node_type, aten=True):
    """ Add a graph operation to a specific NodeType

    Args:
        name (str): Name of the operation in the graph
        node_type (NodeType): To which type to add
        aten (bool, optional): Whether to operation is an ATen or ONNX operation in the graph; Default **True**
    """
    if aten:
        node_type_map_aten[node_type].add(name)
    else:
        node_type_map_onnx[node_type].add(name)


# Default maps
node_type_map_aten = {
    NodeType.CONV:          {'_convolution'},
    NodeType.BATCHNORM:     {'batch_norm'},
    NodeType.CONCAT:        {'cat'},
    NodeType.ELEMW_OP:      {'add'},
    NodeType.IGNORE:        {'leaky_relu', 'relu', 'max_pool2d', 'upsample_nearest2d'},
    NodeType.IGNORE_STOP:   {'size'},
}


node_type_map_onnx = {
    NodeType.CONV:          set(),
    NodeType.BATCHNORM:     set(),
    NodeType.CONCAT:        set(),
    NodeType.ELEMW_OP:      set(),
    NodeType.IGNORE:        set(),
    NodeType.IGNORE_STOP:   {'Concat'},
}
