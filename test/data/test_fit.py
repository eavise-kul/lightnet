#
#   Test fit and reverse fit transforms
#   Copyright EAVISE
#

import pytest
import numpy as np
from PIL import Image
import pandas as pd
import brambox as bb
import lightnet.data.transform as tf


@pytest.fixture(scope='module', params=['np', 'pil'])
def image(request):
    def _image_np(width, height, grayscale=False):
        if grayscale:
            return np.zeros([height, width])
        else:
            return np.zeros([height, width, 3])

    def _image_pil(width, height, grayscale=False):
        if grayscale:
            return Image.new('L', (width, height))
        else:
            return Image.new('RGB', (width, height))

    if (request.param == 'np'):
        return _image_np
    else:
        return _image_pil


@pytest.fixture(scope='module')
def boxes():
    def _boxes(width, height):
        return bb.util.from_dict({
            'image': ['0', '0'],
            'class_label': ['.', '.'],
            'x_top_left': [0, width/2],
            'y_top_left': [0, height/2],
            'width': [width/2, width/4],
            'height': [height/2, height/4],
            'occluded': [0.0, 0.0],
            'truncated': [0.0, 0.0],
            'lost': [False, False],
            'difficult': [False, False],
            'ignore': [False, False],
        })
    return _boxes


def assert_img_size(img, width, height):
    if isinstance(img, np.ndarray):
        assert img.shape[0] == height
        assert img.shape[1] == width
    else:
        w, h = img.size
        assert w == width
        assert h == height


@pytest.mark.parametrize('mode', [True, False])
def test_crop(image, boxes, mode):
    img = image(200, 200, mode)
    df = boxes(200, 200)

    # Width
    img_tf, df_tf = tf.Crop.apply(img, df, dimension=(50, 100), center=True)
    assert_img_size(img_tf, 50, 100)
    assert list(df_tf.x_top_left) == [-25, 25]
    assert list(df_tf.y_top_left) == [0, 50]
    assert list(df_tf.width) == [50, 25]
    assert list(df_tf.height) == [50, 25]

    # Height
    img_tf, df_tf = tf.Crop.apply(img, df, dimension=(100, 50), center=True)
    assert_img_size(img_tf, 100, 50)
    assert list(df_tf.x_top_left) == [0, 50]
    assert list(df_tf.y_top_left) == [-25, 25]
    assert list(df_tf.width) == [50, 25]
    assert list(df_tf.height) == [50, 25]


def test_crop_anno_crop(image, boxes):
    img = image(200, 200, False)
    df = boxes(200, 200)

    img_tf, df_tf = tf.Crop.apply(img, df, dimension=(50, 100), center=True, crop_anno=True)
    assert list(df_tf.x_top_left) == [0, 25]
    assert list(df_tf.y_top_left) == [0, 50]
    assert list(df_tf.width) == [25, 25]
    assert list(df_tf.height) == [50, 25]


def test_crop_anno_intersection(image, boxes):
    img = image(200, 200, False)
    df = boxes(200, 200)

    # Area
    img_tf, df_tf = tf.Crop.apply(img, df, dimension=(50, 100), center=True, intersection_threshold=0.6)
    assert list(df_tf.x_top_left) == [25]
    assert list(df_tf.y_top_left) == [50]
    assert list(df_tf.width) == [25]
    assert list(df_tf.height) == [25]

    # Width
    img_tf, df_tf = tf.Crop.apply(img, df, dimension=(50, 100), center=True, intersection_threshold=(0.6, 0.1))
    assert list(df_tf.x_top_left) == [25]
    assert list(df_tf.y_top_left) == [50]
    assert list(df_tf.width) == [25]
    assert list(df_tf.height) == [25]

    # Height
    img_tf, df_tf = tf.Crop.apply(img, df, dimension=(50, 100), center=True, intersection_threshold=(0.1, 0.6))
    assert list(df_tf.x_top_left) == [-25, 25]
    assert list(df_tf.y_top_left) == [0, 50]
    assert list(df_tf.width) == [50, 25]
    assert list(df_tf.height) == [50, 25]


def test_reverse_crop(image, boxes):
    img = image(200, 200, False)
    df = boxes(200, 200)

    # Width
    img_tf, df_tf = tf.Crop.apply(img, df, dimension=(50, 100), center=True)
    df_rev = tf.ReverseCrop.apply(df_tf, network_size=(50, 100), image_size=(200, 200))
    pd.testing.assert_frame_equal(df, df_rev)

    # Height
    img_tf, df_tf = tf.Crop.apply(img, df, dimension=(100, 50), center=True)
    df_rev = tf.ReverseCrop.apply(df_tf, network_size=(100, 50), image_size=(200, 200))
    pd.testing.assert_frame_equal(df, df_rev)


@pytest.mark.parametrize('mode', [True, False])
def test_letterbox(image, boxes, mode):
    img = image(200, 200, mode)
    df = boxes(200, 200)

    # Width
    img_tf, df_tf = tf.Letterbox.apply(img, df, dimension=(50, 100))
    assert_img_size(img_tf, 50, 100)
    assert list(df_tf.x_top_left) == [0, 25]
    assert list(df_tf.y_top_left) == [25, 50]
    assert list(df_tf.width) == [25, 12.5]
    assert list(df_tf.height) == [25, 12.5]

    # Height
    img_tf, df_tf = tf.Letterbox.apply(img, df, dimension=(100, 50))
    assert_img_size(img_tf, 100, 50)
    assert list(df_tf.x_top_left) == [25, 50]
    assert list(df_tf.y_top_left) == [0, 25]
    assert list(df_tf.width) == [25, 12.5]
    assert list(df_tf.height) == [25, 12.5]


def test_reverse_letterbox(image, boxes):
    img = image(200, 200, False)
    df = boxes(200, 200)

    # Width
    img_tf, df_tf = tf.Crop.apply(img, df, dimension=(50, 100))
    df_rev = tf.ReverseCrop.apply(df_tf, network_size=(50, 100), image_size=(200, 200))
    pd.testing.assert_frame_equal(df, df_rev)

    # Height
    img_tf, df_tf = tf.Crop.apply(img, df, dimension=(100, 50))
    df_rev = tf.ReverseCrop.apply(df_tf, network_size=(100, 50), image_size=(200, 200))
    pd.testing.assert_frame_equal(df, df_rev)


@pytest.mark.parametrize('mode', [True, False])
def test_pad(image, boxes, mode):
    img = image(195, 230, mode)
    df = boxes(195, 230)

    img_tf, df_tf = tf.Pad.apply(img, df, dimension=(100, 50))  # Dimension if multiple
    assert_img_size(img_tf, 200, 250)
    assert list(df_tf.x_top_left) == [2, 99.5]
    assert list(df_tf.y_top_left) == [10, 125]
    assert list(df_tf.width) == [97.5, 48.75]
    assert list(df_tf.height) == [115, 57.5]


def test_reverse_pad(image, boxes):
    img = image(195, 230, False)
    df = boxes(195, 230)

    img_tf, df_tf = tf.Pad.apply(img, df, dimension=(50, 100))
    df_rev = tf.ReversePad.apply(df_tf, network_factor=(50, 100), image_size=(195, 230))
    pd.testing.assert_frame_equal(df, df_rev)
