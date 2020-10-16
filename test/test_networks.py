#
#   Test if network forward function runs
#   Copyright EAVISE
#

import inspect
import pytest
import torch
import lightnet as ln

classification_networks = ['Darknet', 'Darknet19', 'Darknet53', 'MobileDarknet19', 'MobilenetV1', 'MobilenetV2']
anchor_detection_networks = ['DYolo', 'MobilenetYolo', 'MobileYoloV2', 'MobileYoloV2Upsample', 'TinyYoloV2', 'TinyYoloV3', 'YoloV2', 'YoloV2Upsample', 'YoloV3', 'Yolt']
corner_detection_networks = ['Cornernet', 'CornernetSqueeze']
special_networks = ['YoloFusion']


@pytest.fixture(scope='module')
def input_tensor():
    def _input_tensor(dimension, channels=3, batch=1):
        return torch.rand(batch, channels, dimension, dimension)
    return _input_tensor


# Base classification networks
@pytest.mark.parametrize('network', classification_networks)
def test_classification_cpu(network, input_tensor):
    uut = getattr(ln.models, network)(1000).eval()
    it = input_tensor(uut.inner_stride)

    output_tensor = uut(it)
    assert output_tensor.dim() == 2
    assert output_tensor.shape[0] == it.shape[0]
    assert output_tensor.shape[1] == uut.num_classes


@pytest.mark.parametrize('network', classification_networks)
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_classification_cuda(network, input_tensor):
    uut = getattr(ln.models, network)(1000).to('cuda')
    it = input_tensor(uut.inner_stride, batch=2).to('cuda')

    output_tensor = uut(it)
    assert output_tensor.dim() == 2
    assert output_tensor.shape[0] == it.shape[0]
    assert output_tensor.shape[1] == uut.num_classes


# Anchor detection networks
@pytest.mark.parametrize('network', anchor_detection_networks)
def test_anchor_detection_cpu(network, input_tensor):
    uut = getattr(ln.models, network)(20).eval()
    it = input_tensor(uut.inner_stride)

    output_tensor = uut(it)
    if isinstance(output_tensor, torch.Tensor):
        assert output_tensor.dim() == 4
        assert output_tensor.shape[0] == it.shape[0]
        assert output_tensor.shape[1] == len(uut.anchors) * (5 + uut.num_classes)
        assert output_tensor.shape[2] == it.shape[2] // uut.stride
        assert output_tensor.shape[3] == it.shape[3] // uut.stride
    else:
        for i, tensor in enumerate(output_tensor):
            assert tensor.dim() == 4
            assert tensor.shape[0] == it.shape[0]
            assert tensor.shape[1] == len(uut.anchors[i]) * (5 + uut.num_classes)
            assert tensor.shape[2] == it.shape[2] // uut.stride[i]
            assert tensor.shape[3] == it.shape[3] // uut.stride[i]


@pytest.mark.parametrize('network', anchor_detection_networks)
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_anchor_detection_cuda(network, input_tensor):
    uut = getattr(ln.models, network)(20).to('cuda')
    it = input_tensor(uut.inner_stride, batch=2).to('cuda')

    output_tensor = uut(it)
    if isinstance(output_tensor, torch.Tensor):
        assert output_tensor.dim() == 4
        assert output_tensor.shape[0] == it.shape[0]
        assert output_tensor.shape[1] == len(uut.anchors) * (5 + uut.num_classes)
        assert output_tensor.shape[2] == it.shape[2] // uut.stride
        assert output_tensor.shape[3] == it.shape[3] // uut.stride
    else:
        for i, tensor in enumerate(output_tensor):
            assert tensor.dim() == 4
            assert tensor.shape[0] == it.shape[0]
            assert tensor.shape[1] == len(uut.anchors[i]) * (5 + uut.num_classes)
            assert tensor.shape[2] == it.shape[2] // uut.stride[i]
            assert tensor.shape[3] == it.shape[3] // uut.stride[i]


# Corner detection networks
@pytest.mark.parametrize('network', corner_detection_networks)
def test_corner_detection_cpu(network, input_tensor):
    uut = getattr(ln.models, network)(20).eval()
    it = input_tensor(uut.inner_stride)

    output_tensor = uut(it)
    if isinstance(output_tensor, torch.Tensor):
        assert output_tensor.dim() == 4
        assert output_tensor.shape[0] == it.shape[0]
        assert output_tensor.shape[1] == (uut.num_classes + 3) * 2
        assert output_tensor.shape[2] == it.shape[2] // uut.stride
        assert output_tensor.shape[3] == it.shape[3] // uut.stride
    else:
        for tensor in output_tensor:
            assert tensor.dim() == 4
            assert tensor.shape[0] == it.shape[0]
            assert tensor.shape[1] == (uut.num_classes + 3) * 2
            assert tensor.shape[2] == it.shape[2] // uut.stride
            assert tensor.shape[3] == it.shape[3] // uut.stride


@pytest.mark.parametrize('network', corner_detection_networks)
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_corner_detection_cuda(network, input_tensor):
    uut = getattr(ln.models, network)(20).to('cuda')
    it = input_tensor(uut.inner_stride, batch=2).to('cuda')

    output_tensor = uut(it)
    if isinstance(output_tensor, torch.Tensor):
        assert output_tensor.dim() == 4
        assert output_tensor.shape[0] == it.shape[0]
        assert output_tensor.shape[1] == (uut.num_classes + 3) * 2
        assert output_tensor.shape[2] == it.shape[2] // uut.stride
        assert output_tensor.shape[3] == it.shape[3] // uut.stride
    else:
        for tensor in output_tensor:
            assert tensor.dim() == 4
            assert tensor.shape[0] == it.shape[0]
            assert tensor.shape[1] == (uut.num_classes + 3) * 2
            assert tensor.shape[2] == it.shape[2] // uut.stride
            assert tensor.shape[3] == it.shape[3] // uut.stride


# YoloFusion
def test_yolofusion_cpu(input_tensor):
    it = input_tensor(ln.models.YoloFusion.inner_stride, 4)

    for fusion in (0, 1, 10, 22, 27):
        uut = ln.models.YoloFusion(20, fuse_layer=fusion).eval()
        output_tensor = uut(it)
        assert output_tensor.dim() == 4
        assert output_tensor.shape[0] == it.shape[0]
        assert output_tensor.shape[1] == len(uut.anchors) * (5 + uut.num_classes)
        assert output_tensor.shape[2] == it.shape[2] // uut.stride
        assert output_tensor.shape[3] == it.shape[3] // uut.stride


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_yolofusion_cuda(input_tensor):
    it = input_tensor(ln.models.YoloFusion.inner_stride * 2, 4).to('cuda')

    for fusion in (0, 1, 10, 22, 27):
        uut = ln.models.YoloFusion(20, fuse_layer=fusion).to('cuda')
        output_tensor = uut(it)
        assert output_tensor.dim() == 4
        assert output_tensor.shape[0] == it.shape[0]
        assert output_tensor.shape[1] == len(uut.anchors) * (5 + uut.num_classes)
        assert output_tensor.shape[2] == it.shape[2] // uut.stride
        assert output_tensor.shape[3] == it.shape[3] // uut.stride


# All networks tested?
def test_all_networks_tested():
    networks = [
        net for net in dir(ln.models)
        if (inspect.isclass(getattr(ln.models, net)))
        and (issubclass(getattr(ln.models, net), torch.nn.Module))
    ]

    tested_networks = set(
        anchor_detection_networks
        + corner_detection_networks
        + classification_networks
        + special_networks
    )
    for net in networks:
        if net not in tested_networks:
            raise NotImplementedError(f'Network [{net}] is not being tested!')
