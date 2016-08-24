# -*- coding: utf-8 -*-

import pytest
import numpy as np
from audiotrans_transform_onset import OnsetDetectionTransform


def test_accept_arg_of_verbose():
    OnsetDetectionTransform(['-v'])  # no error would be raised


@pytest.mark.parametrize('frame_threshold, feature_threshold, stft_matrices, expecteds', [
    [1, 100,
     [np.repeat(m, 513, axis=0) for m in [
         [[0, 0, 0, 100]],      # 0 (4 * 256 / 44100) = 23ms
         [[150, 200, 50, 20]],  # 1 (8 * 256 / 44100) = 46ms
         [[0, 0, 0, 0]],        # 2 (8 * 256 / 44100) = 69ms
         [[0, 0, 0, 0]]         # 3 (8 * 256 / 44100) = 92ms
     ]],
     [
         [[4, 100 ** 2 - 0 ** 2]],
         [[5, 150 ** 2 - 100 ** 2], [6, 200 ** 2 - 150 ** 2]],
         [],
         []
     ]],

    # [3, 100,
    #  [np.repeat(m, 513, axis=0) for m in [
    #      [[0, 0, 0, 100]],      # 0 (4 * 256 / 44100) = 23ms
    #      [[150, 0, 50, 20]],  # 1 (8 * 256 / 44100) = 46ms
    #      [[0, 0, 0, 0]],        # 2 (8 * 256 / 44100) = 69ms
    #      [[0, 0, 0, 0]]         # 3 (8 * 256 / 44100) = 92ms
    #  ]],
    #  [
    #      [],
    #      [[5, 150 ** 2 - 100 ** 2], [7, 50 ** 2 - 0 ** 2]],
    #      [],
    #      []
    #  ]],

])
def test_repeatedly_transform_should_be_connected_smoothly(frame_threshold,
                                                           feature_threshold,
                                                           stft_matrices,
                                                           expecteds):
    # convert expeced frame to expeced time
    mspf = 256 / 44100 * 1000
    time_threshold = mspf * frame_threshold
    # expecteds = [e if len(e) == 0 else [[a[0] * mspf, a[1]] for a in e] for e in expecteds]
    expecteds = [np.empty((0, 2)) if len(e) == 0 else np.array(e) * (mspf, 1) for e in expecteds]

    tr = OnsetDetectionTransform('-T {} -F {}'.format(time_threshold, feature_threshold).split())

    for i, (stft_matrix, expected) in enumerate(zip(stft_matrices, expecteds)):
        result = tr.transform(stft_matrix)
        # print(i, result, expected)
        # print(i, result.shape, expected.shape)
        # print(result == expected)
#        assert (result == expected).all()

    assert False
