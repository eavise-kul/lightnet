#
#   Base pruning method class
#   Copyright EAVISE
#

from abc import ABC, abstractmethod
import logging
import torch
from ..dependency import get_dependency_map, NodeType

__all__ = ['Pruner']
log = logging.getLogger(__name__)


def default_get_parameters(model):
    return model.parameters()


class Pruner(ABC):
    """ TODO """
    def __init__(self, model, optimizer, input_dimensions, manner="hard", get_parameters=None):
        self.model = model
        self.optimizer = optimizer
        self.manner = manner
        self.dependencies = get_dependency_map(model, input_dimensions)

        if get_parameters is not None:
            self.get_parameters = get_parameters
        else:
            self.get_parameters = default_get_parameters

    def __call__(self, percentage):
        """ TODO """
        with torch.no_grad():
            # Delete gradients
            for param in self.model.parameters():
                if hasattr(param, 'grad') and param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            for buf in self.model.buffers():
                if hasattr(buf, 'grad') and buf.grad is not None:
                    buf.grad.detach_()
                    buf.grad = None

            # Prune
            if percentage > 1:
                percentage /= 100

            if self.manner == "soft":
                prune_count = self.prune(percentage, self.soft_prune)
            else:
                prune_count = self.prune(percentage, self.hard_prune)
                self._update_optimizer()

            return prune_count

    @abstractmethod
    def prune(self, percentage, prune_manner):
        """ TODO
        Note : Should return number of pruned filters
        """
        pass

    def soft_prune(self, conv_node, filter_list, chain=False):
        """ TODO """
        # Set output tensors to zero; filter_list does not get modified
        if conv_node.module.groups != 1:
            raise NotImplementedError('Grouped Convolution')

        mask = torch.zeros(conv_node.module.weight.shape[0], dtype=bool)
        mask.scatter_(0, torch.tensor(filter_list), True)

        conv_node.module.weight[mask] = 0
        if conv_node.module.bias is not None:
            conv_node.module.bias[mask] = 0

        # Chain modifications
        if chain:
            for c in conv_node.children:
                self._soft_chain(c, filter_list, conv_node)

    def hard_prune(self, conv_node, filter_list, chain=True):
        """ TODO """
        # Remove filter_list from output tensor; filter_list does not get modified
        if conv_node.module.groups != 1:
            raise NotImplementedError('Grouped Convolution')

        mask = torch.ones(conv_node.module.weight.shape[0], dtype=bool)
        mask.scatter_(0, torch.tensor(filter_list), False)

        conv_node.module.weight = torch.nn.Parameter(conv_node.module.weight[mask])
        if conv_node.module.bias is not None:
            conv_node.module.bias = torch.nn.Parameter(conv_node.module.bias[mask])
        conv_node.module.out_channels -= len(filter_list)

        # Chain modifications
        if chain:
            for c in conv_node.children:
                self._hard_chain(c, filter_list, conv_node)

    def _soft_chain(self, node, filter_list, parent=None):
        if node.type is NodeType.CONV:
            # Set input tensors to zero; filter_list does not get modified
            if node.module.groups != 1:
                # DW-separable
                mask = torch.zeros(node.module.weight.shape[0], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), True)
                node.module.weight[mask] = 0
            else:
                mask = torch.zeros(node.module.weight.shape[1], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), True)
                node.module.weight[:, mask] = 0
        elif node.type is NodeType.BATCHNORM:
            # Set tensors to zero; filter_list does not get modified
            if node.module.weight is not None:
                mask = torch.zeros(node.module.weight.shape[0], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), True)
                node.module.weight[mask] = 0
                if node.module.bias is not None:
                    node.module.bias[mask] = 0

            if node.module.running_mean is not None:
                mask = torch.zeros(node.module.running_mean.shape[0], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), True)
                node.module.running_mean[mask] = 0
                if node.module.running_var is not None:
                    node.module.running_var[mask] = 0
        elif node.type is CONCAT:
            # Modify filter list according to concat operation
            for p in node.parents:
                if p == parent:
                    break
                else:
                    num = self._get_num_channels(p)
                    filter_list = [f+num for f in filter_list]

        # Recurse
        for c in node.children:
            self._soft_chain(c, filter_list, node)

    def _hard_chain(self, node, filter_list, parent=None):
        if node.type is NodeType.CONV:
            # Remove filter_list from input tensor; filter_list does not get modified
            if node.module.groups != 1:
                # DW-separable
                mask = torch.ones(node.module.weight.shape[0], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), False)
                node.module.weight = torch.nn.Parameter(node.module.weight[mask])
                node.module.in_channels -= len(filter_list)
                node.module.out_channels -= len(filter_list)
                node.module.groups -= len(filter_list)
            else:
                mask = torch.ones(node.module.weight.shape[1], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), False)
                node.module.weight = torch.nn.Parameter(node.module.weight[:, mask])
                node.module.in_channels -= len(filter_list)
        elif node.type is NodeType.BATCHNORM:
            # Remove filter_list from tensor; filter_list does not get modified
            if node.module.weight is not None:
                mask = torch.ones(node.module.weight.shape[0], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), False)
                node.module.weight = torch.nn.Parameter(node.module.weight[mask])
                if node.module.bias is not None:
                    node.module.bias = torch.nn.Parameter(node.module.bias[mask])

            if node.module.running_mean is not None:
                mask = torch.ones(node.module.running_mean.shape[0], dtype=bool)
                mask.scatter_(0, torch.tensor(filter_list), False)
                node.module.running_mean = node.module.running_mean[mask]
                if node.module.running_var is not None:
                    node.module.running_var = node.module.running_var[mask]

            node.module.num_features -= len(filter_list)
        elif node.type is CONCAT:
            # Modify filter list according to concat operation
            for p in node.parents:
                if p == parent:
                    break
                else:
                    num = self._get_num_channels(p)
                    filter_list = [f+num for f in filter_list]

        # Recurse
        for c in node.children:
            self._hard_chain(c, filter_list, node)

    def _get_num_channels(self, node):
        if node.type in {NodeType.CONV, NodeType.BATCHNORM}:
            return node.module.weight.shape[1]
        if node.type is NodeType.CONCAT:
            return sum(self._get_num_channels(p) for p in parents)
        if node.type is NodeType.ELEMW_OP:
            return self._get_num_channels(node.parents[0])

    def _update_optimizer(self):
        if self.optimizer is not None:
            param_groups = list(self.get_parameters(self.model))

            if not isinstance(param_groups[0], dict):
                param_groups = [{'params': param_groups}]

            for old, new in zip(self.optimizer.param_groups, param_groups):
                old.update(new)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        get_param = self.get_parameters.__name__ if hasattr(self.get_parameters, '__name__') else self.get_parameters.__class__.__name__
        return f'{self.__class__.__name__} ({self.model.__class__.__name__}, manner={self.manner}, get_parameters={get_param})'
