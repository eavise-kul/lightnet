#
#   Test fit and reverse fit transforms
#   Copyright EAVISE
#

import pytest
import numpy as np
from PIL import Image
import torch
import pandas as pd
import brambox as bb
import lightnet.data.transform as tf


@pytest.fixture(scope='module', params=['np', 'pil', 'torch'])
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

    def _image_torch(width, height, grayscale=False):
        if grayscale:
            return torch.zeros([height, width])
        else:
            return torch.zeros([3, height, width])

    if (request.param == 'np'):
        return _image_np
    elif (request.param == 'pil'):
        return _image_pil
    else:
        return _image_torch


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
    elif isinstance(img, torch.Tensor):
        assert img.shape[-2] == height
        assert img.shape[-1] == width
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


@pytest.mark.parametrize('mode', [True, False])
def test_fitanno(image, boxes, mode):
    img = image(200, 200, mode)
    df = boxes(300, 300)
    df = df.append({
        'image': '0',
        'class_label': '.',
        'x_top_left': 200,
        'y_top_left': 200,
        'width': 50,
        'height': 100,
        'occluded': 0.0,
        'truncated': 0.0,
        'lost': False,
        'difficult': False,
        'ignore': False,
    }, ignore_index=True)
    df.loc[0, 'x_top_left'] = -50

    _, df_tf = tf.FitAnno.apply(img, df)
    assert list(df_tf.x_top_left) == [0, 150]
    assert list(df_tf.y_top_left) == [0, 150]
    assert list(df_tf.width) == [100, 50]
    assert list(df_tf.height) == [150, 50]


def test_fitanno_filter(image, boxes):
    img = image(200, 200, False)
    df = boxes(200, 200)
    df.loc[0, 'x_top_left'] = -50
    df.loc[0, 'width'] = 100

    # Filter area
    _, df_tf = tf.FitAnno.apply(img, df, crop=False, filter_threshold=0.6)
    assert len(df_tf.index) == 1
    assert list(df_tf.x_top_left) == [100]

    # Filter width/height
    _, df_tf = tf.FitAnno.apply(img, df, crop=False, filter_threshold=(0.6, 0.2))
    assert len(df_tf.index) == 1
    assert list(df_tf.x_top_left) == [100]

    _, df_tf = tf.FitAnno.apply(img, df, crop=False, filter_threshold=(0.2, 0.6))
    assert len(df_tf.index) == 2
    assert list(df_tf.x_top_left) == [-50, 100]

    # Filter ignore
    _, df_tf = tf.FitAnno.apply(img, df, crop=False, filter_threshold=0.6, filter_type='ignore')
    assert len(df_tf.index) == 2
    assert list(df_tf.x_top_left) == [-50, 100]
    assert list(df_tf.ignore) == [True, False]

    # Remove Empty
    df = df.append({
        'image': '0',
        'class_label': '.',
        'x_top_left': 200,
        'y_top_left': 200,
        'width': 0,
        'height': 0,
        'occluded': 0.0,
        'truncated': 0.0,
        'lost': False,
        'difficult': False,
        'ignore': False,
    }, ignore_index=True)

    _, df_tf = tf.FitAnno.apply(img, df, crop=False, filter_threshold=0.6, filter_type='ignore')
    assert len(df_tf.index) == 2
    assert list(df_tf.x_top_left) == [-50, 100]
    assert list(df_tf.ignore) == [True, False]


def test_fitanno_crop(image, boxes):
    img = image(200, 200, False)
    df = boxes(200, 200)
    df.loc[0, 'x_top_left'] = -50
    df.loc[0, 'width'] = 100
    df.loc[1, 'width'] = 250

    # Crop
    _, df_tf = tf.FitAnno.apply(img, df, filter=False)
    assert list(df_tf.x_top_left) == [0, 100]
    assert list(df_tf.y_top_left) == [0, 100]
    assert list(df_tf.width) == [50, 100]
    assert list(df_tf.height) == [100, 50]

    # Remove Empty
    df = df.append({
        'image': '0',
        'class_label': '.',
        'x_top_left': 200,
        'y_top_left': 200,
        'width': 50,
        'height': 100,
        'occluded': 0.0,
        'truncated': 0.0,
        'lost': False,
        'difficult': False,
        'ignore': False,
    }, ignore_index=True)
    print(df[['x_top_left', 'y_top_left', 'width', 'height']])

    _, df_tf = tf.FitAnno.apply(img, df, filter=False)
    assert len(df_tf.index) == 2
    assert list(df_tf.x_top_left) == [0, 100]
