# -*- coding: utf-8 -*-

import pytest
import wave
import numpy as np
from audiotrans_transform_stft import STFTTransform
from audiotrans_transform_onset import (OnsetDetectionTransform,
                                        get_spectral_flux,
                                        gen_local_mean_threshold,
                                        gen_local_maximum_threshold,
                                        gen_exponential_decay_threshold)


def test_accept_arg_of_verbose():
    OnsetDetectionTransform(['-v'])  # no error would be raised


@pytest.mark.parametrize('buf_size,'
                         'local_mean_time,'
                         'local_maximum_time',
                         [[1024, 50, 20],
                          [1024, 100, 20],
                          [1024, 200, 20],
                          [1024, 50, 200],
                          [1024, 100, 200],
                          [1024, 200, 200],
                          [2048, 50, 20],
                          [2048, 100, 20],
                          [2048, 200, 20]])
def test_generated_thresholdeds_should_be_same_with_on_batch(buf_size,
                                                             local_mean_time,
                                                             local_maximum_time):
    w = wave.open('tests/fixture/drums+bass.wav')
    x = np.fromstring(w.readframes(w.getnframes()), np.int16)

    framerate = w.getframerate()
    mspf = 256 / framerate * 1000
    local_mean_frame = int(local_mean_time / mspf)
    local_maximum_frame = int(local_maximum_time / mspf)

    # exercise on batch
    stft_tr = STFTTransform()
    stft_matrix = np.zeros((513, 0))
    for i in range(0, len(x), buf_size):
        tmp_stft_matrix = stft_tr.transform(x[i:i + buf_size])
        stft_matrix = np.concatenate((stft_matrix, tmp_stft_matrix), 1)
    spectrogram = np.abs(stft_matrix) ** 2
    e_spectral_flux = get_spectral_flux(spectrogram)
    e_local_mean_threshold = gen_local_mean_threshold(e_spectral_flux, local_mean_frame, 1.5)
    e_local_maximum_threshold = gen_local_maximum_threshold(e_spectral_flux, local_maximum_frame)
    e_exponential_decay_threshold = gen_exponential_decay_threshold(e_spectral_flux, 0.8)

    # exercise on stream
    stft_tr = STFTTransform()
    onset_tr = OnsetDetectionTransform('-r {} -p 1.5 -m {} -M {} -d 0.8 -D'
                                       .format(framerate, local_mean_time, local_maximum_time)
                                       .split())

    spectral_flux = np.zeros(0)
    local_mean_threshold = np.zeros(0)
    local_maximum_threshold = np.zeros(0)
    exponential_decay_threshold = np.zeros(0)

    for i in range(0, len(x), buf_size):
        tmp_stft_matrix = stft_tr.transform(x[i:i + buf_size])
        tmp_results = onset_tr.transform(tmp_stft_matrix)
        spectral_flux = np.append(spectral_flux, tmp_results[0])
        local_mean_threshold = np.append(local_mean_threshold, tmp_results[1])
        local_maximum_threshold = np.append(local_maximum_threshold, tmp_results[2])
        exponential_decay_threshold = np.append(exponential_decay_threshold, tmp_results[3])

    assert (
        spectral_flux.shape ==
        local_mean_threshold.shape ==
        local_maximum_threshold.shape ==
        exponential_decay_threshold.shape), \
        "each thresholds should have same form."

    l = len(local_mean_threshold)
    assert (np.isclose(spectral_flux, e_spectral_flux[:l])).all(), \
        "same results between batch and transform"
    assert (np.isclose(local_mean_threshold, e_local_mean_threshold[:l])).all(), \
        "same results between batch and transform"
    assert (np.isclose(local_maximum_threshold, e_local_maximum_threshold[:l])).all(), \
        "same results between batch and transform"
    assert (np.isclose(exponential_decay_threshold, e_exponential_decay_threshold[:l])).all(), \
        "same results between batch and transform"


@pytest.mark.parametrize('buf_size,'
                         'local_mean_time,'
                         'local_mean_multiplier,'
                         'local_maximum_time,'
                         'decay_factor',
                         [[1024, 50,  1.5, 20,  0.9],
                          [1024, 100, 1.5, 20,  0.9],
                          [1024, 200, 1.5, 20,  0.8],
                          [1024, 50,  1.5, 200, 0.8],
                          [1024, 100, 1.5, 200, 0.8],
                          [1024, 200, 1.5, 200, 0.8],
                          [2048, 50,  1.5, 20,  0.8],
                          [2048, 100, 1.5, 20,  0.8],
                          [2048, 200, 1.5, 20,  0.8]])
def test_generated_thresholded_flux_should_be_same_with_on_batch(buf_size,
                                                                 local_mean_time,
                                                                 local_mean_multiplier,
                                                                 local_maximum_time,
                                                                 decay_factor):
    mspf = 256 / 44100 * 1000
    local_mean_frame = int(local_mean_time / mspf)
    local_maximum_frame = int(local_maximum_time / mspf)

    w = wave.open('tests/fixture/drums+bass.wav')
    x = np.fromstring(w.readframes(w.getnframes()), np.int16)

    # exercise on batch
    stft_tr = STFTTransform()
    stft_matrix = np.zeros((513, 0))
    for i in range(0, len(x), buf_size):
        tmp_stft_matrix = stft_tr.transform(x[i:i + buf_size])
        stft_matrix = np.concatenate((stft_matrix, tmp_stft_matrix), 1)
    spectrogram = np.abs(stft_matrix) ** 2
    e_spectral_flux = get_spectral_flux(spectrogram)
    e_local_mean_threshold = gen_local_mean_threshold(e_spectral_flux,
                                                      local_mean_frame,
                                                      local_mean_multiplier)
    e_local_maximum_threshold = gen_local_maximum_threshold(e_spectral_flux, local_maximum_frame)
    e_exponential_decay_threshold = gen_exponential_decay_threshold(e_spectral_flux, decay_factor)
    e_isonsets = ((e_spectral_flux - e_local_mean_threshold >= 0) *
                  (e_spectral_flux - e_local_maximum_threshold >= 0) *
                  (e_spectral_flux - e_exponential_decay_threshold >= 0))
    e_onsets = np.where(e_isonsets)[0] * mspf

    # exercise on stream
    stft_tr = STFTTransform()
    onset_tr = OnsetDetectionTransform('-p {} -m {} -M {} -d {}'
                                       .format(local_mean_multiplier,
                                               local_mean_time,
                                               local_maximum_time,
                                               decay_factor).split())

    result = np.zeros(0)
    for i in range(0, len(x), buf_size):
        tmp_stft_matrix = stft_tr.transform(x[i:i + buf_size])
        tmp_result = onset_tr.transform(tmp_stft_matrix)
        result = np.append(result, tmp_result)

    l = len(result)

    # verify
    assert np.allclose(result, e_onsets[:l]), \
        "same results between batch and transform"
