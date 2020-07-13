#
#   Test NMS implementations
#   Copyright EAVISE
#

import pytest
import torch
import pandas as pd
import brambox as bb
import lightnet.data.transform as tf


@pytest.fixture(scope='module')
def boxes():
    return [
        [1, 0,   0, 250, 250, 0.5, 1],      # Box E : Other image | IoU(A,E) = 1
        [0, 0,   0, 250, 250, 0.6, 1],      # Box D : Other class | IoU(A,D) = 1
        [0, 200, 0, 450, 250, 0.7, 0],      # Box C : IoU(A,C) < 0.4 | IoU(B,C) > 0.4
        [0, 100, 0, 350, 250, 0.8, 0],      # Box B : IoU(A,B) > 0.4
        [0, 0,   0, 250, 250, 0.9, 0],      # Box A
    ]


def test_nms(boxes):
    input_tensor = torch.tensor(boxes)
    input_pd = tf.TensorToBrambox.apply(input_tensor.clone())
    nms = tf.NMS(0.4)

    # Check tensor output
    out1 = nms(input_tensor)
    assert out1.shape[0] == 4
    assert out1.shape[1] == 7
    assert list(out1[:, 0]) == [1, 0, 0, 0]
    assert list(out1[:, 5]) == [0.5, 0.6, 0.7, 0.9]

    # Compare pandas and torch
    out1_pd = tf.TensorToBrambox.apply(out1).sort_values('confidence').reset_index(drop=True)
    out2 = nms(input_pd).sort_values('confidence').reset_index(drop=True)
    pd.testing.assert_frame_equal(out1_pd, out2)


def test_nms_ignore_class(boxes):
    input_tensor = torch.tensor(boxes)
    input_pd = tf.TensorToBrambox.apply(input_tensor.clone())
    nms = tf.NMS(0.4, class_nms=False)

    # Check tensor output
    out1 = nms(input_tensor)
    assert out1.shape[0] == 3
    assert out1.shape[1] == 7
    assert list(out1[:, 0]) == [1, 0, 0]
    assert list(out1[:, 5]) == [0.5, 0.7, 0.9]

    # Compare pandas and torch
    out1_pd = tf.TensorToBrambox.apply(out1).sort_values('confidence').reset_index(drop=True)
    out2 = nms(input_pd).sort_values('confidence').reset_index(drop=True)
    pd.testing.assert_frame_equal(out1_pd, out2)


def test_nms_fast(boxes):
    input_tensor = torch.tensor(boxes)
    input_pd = tf.TensorToBrambox.apply(input_tensor.clone())
    nms = tf.NMSFast(0.4)

    # Check tensor output
    out1 = nms(input_tensor)
    assert out1.shape[0] == 3
    assert out1.shape[1] == 7
    assert list(out1[:, 0]) == [1, 0, 0]
    assert list(out1[:, 5]) == [0.5, 0.6, 0.9]

    # Compare pandas and torch
    out1_pd = tf.TensorToBrambox.apply(out1).sort_values('confidence').reset_index(drop=True)
    out2 = nms(input_pd).sort_values('confidence').reset_index(drop=True)
    pd.testing.assert_frame_equal(out1_pd, out2)


def test_nms_fast_ignore_class(boxes):
    input_tensor = torch.tensor(boxes)
    input_pd = tf.TensorToBrambox.apply(input_tensor.clone())
    nms = tf.NMSFast(0.4, class_nms=False)

    # Check tensor output
    out1 = nms(input_tensor)
    assert out1.shape[0] == 2
    assert out1.shape[1] == 7
    assert list(out1[:, 0]) == [1, 0]
    assert list(out1[:, 5]) == [0.5, 0.9]

    # Compare pandas and torch
    out1_pd = tf.TensorToBrambox.apply(out1).sort_values('confidence').reset_index(drop=True)
    out2 = nms(input_pd).sort_values('confidence').reset_index(drop=True)
    pd.testing.assert_frame_equal(out1_pd, out2)


def test_nms_soft(boxes):
    input_tensor = torch.tensor(boxes)
    input_pd = tf.TensorToBrambox.apply(input_tensor.clone())
    nms = tf.NMSSoft(0.4)

    # Check tensor output
    out1 = nms(input_tensor)
    assert out1.shape[0] == 5
    assert out1.shape[1] == 7
    assert list(out1[:, 0]) == [1, 0, 0, 0, 0]
    assert list(out1[:, 1]) == [0, 0, 200, 100, 0]
    assert list(out1[:, 3]) == [250, 250, 450, 350, 250]
    # assert list(out1[:, 5]) == []  # TODO : compute scores manually

    # Compare pandas and torch
    out1_pd = tf.TensorToBrambox.apply(out1).sort_values('confidence').reset_index(drop=True)
    out2 = nms(input_pd).sort_values('confidence').reset_index(drop=True)
    pd.testing.assert_frame_equal(out1_pd, out2)


def test_nms_soft_fast_ignore_class(boxes):
    input_tensor = torch.tensor(boxes)
    input_pd = tf.TensorToBrambox.apply(input_tensor.clone())
    nms = tf.NMSSoft(0.4, class_nms=False)

    # Check tensor output
    out1 = nms(input_tensor)
    assert out1.shape[0] == 5
    assert out1.shape[1] == 7
    assert list(out1[:, 0]) == [1, 0, 0, 0, 0]
    assert list(out1[:, 1]) == [0, 0, 200, 100, 0]
    assert list(out1[:, 3]) == [250, 250, 450, 350, 250]
    # assert list(out1[:, 5]) == []  # TODO : compute scores manually

    # Compare pandas and torch
    out1_pd = tf.TensorToBrambox.apply(out1).sort_values('confidence').reset_index(drop=True)
    out2 = nms(input_pd).sort_values('confidence').reset_index(drop=True)
    pd.testing.assert_frame_equal(out1_pd, out2)


def test_nms_soft_fast(boxes):
    input_tensor = torch.tensor(boxes)
    input_pd = tf.TensorToBrambox.apply(input_tensor.clone())
    nms = tf.NMSSoftFast(0.4)

    # Check tensor output
    out1 = nms(input_tensor)
    assert out1.shape[0] == 5
    assert out1.shape[1] == 7
    assert list(out1[:, 0]) == [1, 0, 0, 0, 0]
    assert list(out1[:, 1]) == [0, 0, 200, 100, 0]
    assert list(out1[:, 3]) == [250, 250, 450, 350, 250]
    # assert list(out1[:, 5]) == []  # TODO : compute scores manually

    # Compare pandas and torch
    out1_pd = tf.TensorToBrambox.apply(out1).sort_values('confidence').reset_index(drop=True)
    out2 = nms(input_pd).sort_values('confidence').reset_index(drop=True)
    pd.testing.assert_frame_equal(out1_pd, out2)


def test_nms_soft_fast_ignore_class(boxes):
    input_tensor = torch.tensor(boxes)
    input_pd = tf.TensorToBrambox.apply(input_tensor.clone())
    nms = tf.NMSSoftFast(0.4, class_nms=False)

    # Check tensor output
    out1 = nms(input_tensor)
    assert out1.shape[0] == 5
    assert out1.shape[1] == 7
    assert list(out1[:, 0]) == [1, 0, 0, 0, 0]
    assert list(out1[:, 1]) == [0, 0, 200, 100, 0]
    assert list(out1[:, 3]) == [250, 250, 450, 350, 250]
    # assert list(out1[:, 5]) == []  # TODO : compute scores manually

    # Compare pandas and torch
    out1_pd = tf.TensorToBrambox.apply(out1).sort_values('confidence').reset_index(drop=True)
    out2 = nms(input_pd).sort_values('confidence').reset_index(drop=True)
    pd.testing.assert_frame_equal(out1_pd, out2)
