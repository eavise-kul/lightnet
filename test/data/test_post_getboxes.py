#
#   Test Get*Boxes function on output
#   Note that we only perform some basic sanity checking
#   Copyright EAVISE
#

import pytest
import torch
import lightnet as ln
import lightnet.data.transform as tf

uut = [
    (
        tf.GetDarknetBoxes,
        (0.5, 32, [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]),
        dict(),
        lambda img, cls: (img, 5*(cls+5), 13, 13)
    ),
    (
        tf.GetMultiScaleDarknetBoxes,
        (
            0.5,
            (32, 16, 8),
            # Default YoloV3 anchors divided by stride
            [[(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)], [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)], [(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)]]
        ),
        dict(),
        lambda img, cls: (3, img, 3*(cls+5), 13, 13)
    ),
    (
        tf.GetCornerBoxes,
        (0.5, 0.5, 4, 3),
        dict(),
        lambda img, cls: (img, 2*(cls+3), 128, 128)
    )
]


@pytest.mark.parametrize('data', uut)
@pytest.mark.parametrize('images', [1, 8])
@pytest.mark.parametrize('classes', [1, 20])
def test_getboxes(data, images, classes):
    boxes = data[0](*data[1], **data[2])
    t = torch.rand(data[3](images, classes))

    out = boxes(t)
    assert out.ndim == 2
    assert out.shape[1] == 7
    assert (set(out[:, 0].unique().tolist()) <= set(range(images)))     # batch_num should be between 0-num_images
    assert (out[:, 5] > 0.5).all()                                      # confidence should be bigger than threshold
    assert (set(out[:, 6].unique().tolist()) <= set(range(classes)))    # class_id should be between 0-num_classes


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('data', uut)
@pytest.mark.parametrize('images', [1, 8])
@pytest.mark.parametrize('classes', [1, 20])
def test_getboxes_cuda(data, images, classes):
    boxes = data[0](*data[1], **data[2])
    t = torch.rand(data[3](images, classes), device='cuda')

    out = boxes(t)
    assert out.ndim == 2
    assert out.shape[1] == 7
    assert (set(out[:, 0].unique().tolist()) <= set(range(images)))     # batch_num should be between 0-num_images
    assert (out[:, 5] > 0.5).all()                                      # confidence should be bigger than threshold
    assert (set(out[:, 6].unique().tolist()) <= set(range(classes)))    # class_id should be between 0-num_classes
