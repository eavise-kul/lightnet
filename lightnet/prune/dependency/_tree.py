#
#   Tree data structure and related functionality
#   Copyright EAVISE
#
__all__ = ['Node', 'traverse_depth_first', 'traverse_breadth_first']


class Node:
    """ Tree node.
    This data structure is used to get the dependency tree when pruning a convolution.
    """
    def __init__(self, nodetype, nodename=None, module=None, metadata=None, parents=None, children=None):
        self.type = nodetype
        self.name = nodename
        self.module = module
        self.metadata = metadata
        self.parents = parents if parents is not None else []
        self.children = children if children is not None else []

    def __str__(self):
        return f'Node({self.type.name}, {self.name}, #parents={len(self.parents)}, #children={len(self.children)})'

    def __repr__(self):
        string = f'Node({self.type.name}, {self.name}, parents=['
        string += ', '.join(str(p) for p in self.parents)
        string += '], children=['
        string += ', '.join(str(c) for c in self.children)
        string += ']'

        return string

    def __eq__(self, other):
        if (not isinstance(other, Node)
           or (self.type != other.type)
           or (self.name != other.name)
           or (self.module != other.module)
           or (self.metadata != other.metadata)):
            return False
        return True

    def add_child(self, node):
        self.children.append(node)
        node.parents.append(self)
        return self


def traverse_depth_first(node, yield_root=True, level=0):
    """ Generator that traverses the dependency tree in a depth first manner and returns (level, node) tuples. |br|
    The general gist of this traversal is the following:

    .. code::

        1. Yield node N
        2. Foreach parent P of N (except previous node)
            3. Yield P
            4. Foreach parent PP of P
                5. Recurse to step 3 with P=PP
        6. Foreach child C of N
            7. Recurse to step 1 with C=N (and N=previous)

    Args:
        node (Node): Base node to start from
        yield_root(bool, optional): Whether to start by yielding the root node or not; Default **True**
        level (int, optional): Starting level value; Default **0**

    Returns:
        Generator : generator which returns `(leve, node)` tuples when iterated over

    Note:
        Strictly speaking, this is not depht first, as we yield the children whilst going down the path,
        but this is what we needed for lightnet.
    """
    def _traverse_depth_first(node, yield_root, prev, level):
        if yield_root:
            yield (level, node)
            yield from traverse_depth_first_parents(node, prev, level-1)

        for child in node.children:
            yield from _traverse_depth_first(child, True, [node], level+1)

    yield from _traverse_depth_first(node, yield_root, node.parents, level if yield_root else level-1)


def traverse_depth_first_parents(node, skip=None, level=0):
    """ Parent function of `traverse_depth_first` """
    if skip is None:
        skip = list()

    for p in node.parents:
        if p not in skip:
            yield (level, p)
            yield from traverse_depth_first_parents(p, list(), level-1)


def traverse_breadth_first(node, yield_root=True, level=0):
    """ Generator that traverses the dependency tree in a breadth first manner and returns (level, node) tuples. |br|
    The general gist of this traversal is the following:

    .. code::

        1. Foreach child C of node N
            2. Foreach parent P of C (except N)
                3. Recurse to step 2 with C=P and N=None
            4. Foreach parent P of C (except N)
                5. Yield P
            6. Yield C
        7. Foreach child C of node N
            8. Recurse to step 1 with N=C

    Args:
        node (Node): Base node to start from
        yield_root(bool, optional): Whether to start by yielding the root node or not; Default **True**
        level (int, optional): Starting level value; Default **0**

    Returns:
        Generator : generator which returns `(level, node)` tuples when iterated over
    """
    def _traverse_breadth_first(node, yield_root, level):
        if yield_root:
            yield (level, node)

        for child in node.children:
            yield from traverse_parents_breadth_first(child, [node], level)
            yield (level+1, child)

        for child in node.children:
            yield from traverse_breadth_first(child, False, level+1)

    yield from _traverse_breadth_first(node, yield_root, level if yield_root else level-1)


def traverse_parents_breadth_first(node, skip=None, level=0):
    """ Parent function of `traverse_breadth_first` """
    if skip is None:
        skip = list()

    for p in node.parents:
        if p not in skip:
            yield from traverse_parents_breadth_first(p, list(), level-1)

    for p in node.parents:
        if p not in skip:
            yield (level, p)
