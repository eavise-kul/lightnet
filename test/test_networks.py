#
#   Test if network forward function runs
#   Copyright EAVISE
#

import pytest
import torch
import lightnet as ln

detection_networks = ['Yolo', 'Yolt', 'DYolo', 'TinyYolo', 'MobileNetYolo']
classification_networks = ['Darknet', 'Darknet19']


@pytest.fixture(scope='module')
def input_tensor():
    return torch.rand(1, 3, 416, 416)


# Base classification networks
@pytest.mark.parametrize('network', classification_networks)
def test_classification_cpu(network, input_tensor):
    uut = getattr(ln.models, network)()

    output_tensor = uut(input_tensor)
    assert output_tensor.dim() == 2
    assert output_tensor.shape[0] == 1
    assert output_tensor.shape[1] == uut.num_classes


@pytest.mark.parametrize('network', classification_networks)
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_classification_cuda(network, input_tensor):
    uut = getattr(ln.models, network)().to('cuda')
    input_tensor = input_tensor.to('cuda')

    output_tensor = uut(input_tensor)
    assert output_tensor.dim() == 2
    assert output_tensor.shape[0] == 1
    assert output_tensor.shape[1] == uut.num_classes


# Base detection networks
@pytest.mark.parametrize('network', detection_networks)
def test_detection_cpu(network, input_tensor):
    uut = getattr(ln.models, network)()

    output_tensor = uut(input_tensor)
    assert output_tensor.dim() == 4
    assert output_tensor.shape[0] == 1
    assert output_tensor.shape[1] == len(uut.anchors) * (5 + uut.num_classes)
    assert output_tensor.shape[2] == 416 // uut.stride
    assert output_tensor.shape[3] == 416 // uut.stride


@pytest.mark.parametrize('network', detection_networks)
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_detection_cuda(network, input_tensor):
    uut = getattr(ln.models, network)().to('cuda')
    input_tensor = input_tensor.to('cuda')

    output_tensor = uut(input_tensor)
    assert output_tensor.dim() == 4
    assert output_tensor.shape[0] == 1
    assert output_tensor.shape[1] == len(uut.anchors) * (5 + uut.num_classes)
    assert output_tensor.shape[2] == 416 // uut.stride
    assert output_tensor.shape[3] == 416 // uut.stride


# YoloFusion
def test_yolofusion_cpu():
    input_tensor = torch.rand(1, 4, 416, 416)

    for fusion in (0, 1, 10, 22, 27):
        uut = ln.models.YoloFusion(fuse_layer=fusion)
        output_tensor = uut(input_tensor)
        assert output_tensor.dim() == 4
        assert output_tensor.shape[0] == 1
        assert output_tensor.shape[1] == len(uut.anchors) * (5 + uut.num_classes)
        assert output_tensor.shape[2] == 416 // uut.stride
        assert output_tensor.shape[3] == 416 // uut.stride


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_yolofusion_cuda():
    input_tensor = torch.rand(1, 4, 416, 416).to('cuda')

    for fusion in (0, 1, 10, 22, 27):
        uut = ln.models.YoloFusion(fuse_layer=fusion).to('cuda')
        output_tensor = uut(input_tensor)
        assert output_tensor.dim() == 4
        assert output_tensor.shape[0] == 1
        assert output_tensor.shape[1] == len(uut.anchors) * (5 + uut.num_classes)
        assert output_tensor.shape[2] == 416 // uut.stride
        assert output_tensor.shape[3] == 416 // uut.stride
