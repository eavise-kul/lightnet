#
#   Test pre-processing transforms with multiple different input types
#   Copyright EAVISE
#

import random
import pytest
import torch
import torchvision
import numpy as np
from PIL import Image
import lightnet.data.transform as tf


@pytest.fixture(scope='module')
def image():
    def _image(width, height, grayscale):
        if grayscale:
            img_np = np.random.randint(256, size=(height, width), dtype='uint8')
        else:
            img_np = np.random.randint(256, size=(height, width, 3), dtype='uint8')

        img_pil = Image.fromarray(img_np)
        img_torch = torchvision.transforms.ToTensor()(img_np)

        return img_np, img_pil, img_torch
    return _image


def assert_image_dim_equal(img_np, img_pil, img_torch):
    if img_np is not None and img_torch is not None:
        assert list(img_np.shape[:2]) == list(img_torch.shape[-2:])
    if img_np is not None and img_pil is not None:
        assert list(img_np.shape[:2]) == list([img_pil.height, img_pil.width])
    if img_pil is not None and img_torch is not None:
        assert list([img_pil.height, img_pil.width]) == list(img_torch.shape[-2:])


def assert_image_content_equal(img_np, img_pil, img_torch, assertion=np.testing.assert_array_equal, **kwargs):
    if img_pil is not None:
        img_pil = np.asarray(img_pil).astype('int64')
    if img_torch is not None:
        img_torch = np.asarray(torchvision.transforms.ToPILImage()(img_torch)).astype('int64')
    if img_np is not None:
        img_np = img_np.astype('int64')

    errors = []

    if img_np is not None and img_pil is not None:
        try:
            assertion(img_np, img_pil, err_msg="NumPy and PIL", **kwargs)
        except AssertionError as err:
            errors.append(str(err))

    if img_np is not None and img_torch is not None:
        try:
            assertion(img_np, img_torch, err_msg="NumPy and PyTorch", **kwargs)
        except AssertionError as err:
            errors.append(str(err))

    if img_pil is not None and img_torch is not None:
        try:
            assertion(img_pil, img_torch, err_msg="PIL and PyTorch", **kwargs)
        except AssertionError as err:
            errors.append(str(err))

    if len(errors) > 0:
        raise AssertionError("\n\n".join(errors))


def assert_image_equal(img_np, img_pil, img_torch, assertion=np.testing.assert_array_equal, **kwargs):
    assert_image_dim_equal(img_np, img_pil, img_torch)
    assert_image_content_equal(img_np, img_pil, img_torch, assertion, **kwargs)


@pytest.mark.parametrize('grayscale', [True, False])
def test_crop(image, grayscale):
    img_np, img_pil, img_torch = image(400, 385, grayscale)

    # Crop height
    assert_image_equal(img_np, img_pil, img_torch)
    crop = tf.Crop(dimension=(400, 350))
    tf_np = crop(img_np)
    tf_pil = crop(img_pil)
    tf_torch = crop(img_torch)
    assert list(tf_torch.shape[-2:]) == [350, 400]
    assert_image_equal(tf_np, tf_pil, tf_torch)

    # Crop width
    assert_image_equal(img_np, img_pil, img_torch)
    crop = tf.Crop(dimension=(200, 385))
    tf_np = crop(img_np)
    tf_pil = crop(img_pil)
    tf_torch = crop(img_torch)
    assert list(tf_torch.shape[-2:]) == [385, 200]
    assert_image_equal(tf_np, tf_pil, tf_torch)

    # Shrink - Crop height
    assert_image_equal(img_np, img_pil, img_torch)
    crop = tf.Crop(dimension=(300, 250))
    tf_np = crop(img_np)
    tf_pil = crop(img_pil)
    tf_torch = crop(img_torch)
    assert list(tf_torch.shape[-2:]) == [250, 300]
    assert_image_dim_equal(tf_np, tf_pil, tf_torch)
    assert_image_content_equal(tf_np, None, tf_torch, np.testing.assert_allclose, atol=2)

    # Enlarge - Crop width
    assert_image_equal(img_np, img_pil, img_torch)
    crop = tf.Crop(dimension=(500, 600))
    tf_np = crop(img_np)
    tf_pil = crop(img_pil)
    tf_torch = crop(img_torch)
    assert list(tf_torch.shape[-2:]) == [600, 500]
    assert_image_dim_equal(tf_np, tf_pil, tf_torch)
    assert_image_content_equal(tf_np, None, tf_torch, np.testing.assert_allclose, atol=2)


@pytest.mark.parametrize('grayscale', [True, False])
def test_letterbox(image, grayscale):
    img_np, img_pil, img_torch = image(400, 385, grayscale)

    # Letterbox height
    assert_image_equal(img_np, img_pil, img_torch)
    lb = tf.Letterbox(dimension=(400, 450), fill_color=50)
    tf_np = lb(img_np)
    tf_pil = lb(img_pil)
    tf_torch = lb(img_torch)
    assert list(tf_torch.shape[-2:]) == [450, 400]
    assert_image_equal(tf_np, tf_pil, tf_torch)

    # Letterbox width
    assert_image_equal(img_np, img_pil, img_torch)
    lb = tf.Letterbox(dimension=(500, 385))
    tf_np = lb(img_np)
    tf_pil = lb(img_pil)
    tf_torch = lb(img_torch)
    assert list(tf_torch.shape[-2:]) == [385, 500]
    assert_image_equal(tf_np, tf_pil, tf_torch)

    # Shrink - Letterbox height
    assert_image_equal(img_np, img_pil, img_torch)
    lb = tf.Letterbox(dimension=(250, 300))
    tf_np = lb(img_np)
    tf_pil = lb(img_pil)
    tf_torch = lb(img_torch)
    assert list(tf_torch.shape[-2:]) == [300, 250]
    assert_image_dim_equal(tf_np, tf_pil, tf_torch)
    assert_image_content_equal(tf_np, None, tf_torch, np.testing.assert_allclose, atol=2)

    # Enlarge - Letterbox width
    assert_image_equal(img_np, img_pil, img_torch)
    lb = tf.Letterbox(dimension=(600, 500))
    tf_np = lb(img_np)
    tf_pil = lb(img_pil)
    tf_torch = lb(img_torch)
    assert list(tf_torch.shape[-2:]) == [500, 600]
    assert_image_dim_equal(tf_np, tf_pil, tf_torch)
    assert_image_content_equal(tf_np, None, tf_torch, np.testing.assert_allclose, atol=2)


@pytest.mark.parametrize('grayscale', [True, False])
def test_pad(image, grayscale):
    img_np, img_pil, img_torch = image(400, 385, grayscale)

    # Pad height
    assert_image_equal(img_np, img_pil, img_torch)
    pad = tf.Pad(dimension=(100, 32), fill_color=50)
    tf_np = pad(img_np)
    tf_pil = pad(img_pil)
    tf_torch = pad(img_torch)
    assert list(tf_torch.shape[-2:]) == [416, 400]
    assert_image_equal(tf_np, tf_pil, tf_torch)

    # Pad width
    assert_image_equal(img_np, img_pil, img_torch)
    pad = tf.Pad(dimension=(128, 5))
    tf_np = pad(img_np)
    tf_pil = pad(img_pil)
    tf_torch = pad(img_torch)
    assert list(tf_torch.shape[-2:]) == [385, 512]
    assert_image_equal(tf_np, tf_pil, tf_torch)


@pytest.mark.parametrize('grayscale', [True, False])
def test_flip(image, grayscale):
    img_np, img_pil, img_torch = image(400, 400, grayscale)

    def get_params(tf):
        tf.flip_h = bool(tf.horizontal)
        tf.flip_v = bool(tf.vertical)

    flip = tf.RandomFlip(1, 1)
    flip._get_params = get_params.__get__(flip, tf.RandomFlip)

    # Flip Horizontal
    flip.horizontal = 1
    flip.vertical = 0

    assert_image_equal(img_np, img_pil, img_torch)
    tf_np = flip(img_np)
    tf_pil = flip(img_pil)
    tf_torch = flip(img_torch)
    assert_image_equal(tf_np, tf_pil, tf_torch)

    # Flip Vertical
    flip.horizontal = 0
    flip.vertical = 1

    assert_image_equal(img_np, img_pil, img_torch)
    tf_np = flip(img_np)
    tf_pil = flip(img_pil)
    tf_torch = flip(img_torch)
    assert_image_equal(tf_np, tf_pil, tf_torch)

    # Flip Both
    flip.horizontal = 1
    flip.vertical = 1

    assert_image_equal(img_np, img_pil, img_torch)
    tf_np = flip(img_np)
    tf_pil = flip(img_pil)
    tf_torch = flip(img_torch)
    assert_image_equal(tf_np, tf_pil, tf_torch)


def test_hsv(image):
    img_np, img_pil, img_torch = image(400, 400, False)
    dh = random.uniform(-0.5, 0.5)
    ds = random.uniform(1, 2)
    dv = random.uniform(1, 2)

    def get_params(tf):
        tf.dh = dh
        tf.ds = ds
        tf.dv = dv

    hsv = tf.RandomHSV(None, None, None)
    hsv._get_params = get_params.__get__(hsv, tf.RandomHSV)

    assert_image_equal(img_np, img_pil, img_torch)
    tf_np = hsv(img_np)
    tf_pil = hsv(img_pil)
    tf_torch = hsv(img_torch)
    assert_image_dim_equal(tf_np, tf_pil, tf_torch)
    assert_image_content_equal(tf_np, None, tf_torch, np.testing.assert_allclose, atol=2)


@pytest.mark.parametrize('grayscale', [True, False])
def test_jitter(image, grayscale):
    img_np, img_pil, img_torch = image(400, 400, grayscale)
    crop_left = random.randint(-120, 120)
    crop_right = random.randint(-120, 120)
    crop_top = random.randint(-120, 120)
    crop_bottom = random.randint(-120, 120)

    def get_params(tf, w, h):
        tf.crop = (crop_left, crop_top, w-crop_right, h-crop_bottom)

    jitter = tf.RandomJitter(0.3, fill_color=50)
    jitter._get_params = get_params.__get__(jitter, tf.RandomJitter)

    assert_image_equal(img_np, img_pil, img_torch)
    tf_np = jitter(img_np)
    tf_pil = jitter(img_pil)
    tf_torch = jitter(img_torch)
    assert_image_equal(tf_np, tf_pil, tf_torch)
