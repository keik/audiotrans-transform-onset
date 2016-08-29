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


def test_generated_flux_should_be_same_with_on_batch():
    w = wave.open('tests/fixture/drums+bass.wav')
    x = np.fromstring(w.readframes(w.getnframes()), np.int16)

    # exercise on batch
    stft_tr = STFTTransform()
    stft_matrix = np.zeros((513, 0))
    for i in range(0, len(x), 1024):
        tmp_stft_matrix = stft_tr.transform(x[i:i+1024])
        stft_matrix = np.concatenate((stft_matrix, tmp_stft_matrix), 1)
    spectrogram = np.abs(stft_matrix) ** 2
    flux_on_batch = get_spectral_flux(spectrogram)

    # exercise on stream
    stft_tr = STFTTransform()
    onset_tr = OnsetDetectionTransform(**{'debug_point': 'flux'})
    result = np.zeros(0)
    for i in range(0, len(x), 1024):
        tmp_stft_matrix = stft_tr.transform(x[i:i+1024])
        tmp_flux = onset_tr.transform(tmp_stft_matrix)
        result = np.append(result, tmp_flux)

    # verify
    assert np.allclose(result, flux_on_batch), \
        "same results between batch and transform"


@pytest.mark.parametrize('buf_size, local_mean_time', [
    [1024, 50],
    [1024, 100],
    [1024, 200],
    [2048, 50],
    [2048, 100],
    [2048, 200]
])
def test_generated_thresholded_flux_should_be_same_with_on_batch(buf_size, local_mean_time):
    mspf = 256 / 44100 * 1000
    local_mean_frame = int(local_mean_time / mspf)

    w = wave.open('tests/fixture/drums+bass.wav')
    x = np.fromstring(w.readframes(w.getnframes()), np.int16)

    # exercise on batch
    stft_tr = STFTTransform()
    stft_matrix = np.zeros((513, 0))
    for i in range(0, len(x), buf_size):
        tmp_stft_matrix = stft_tr.transform(x[i:i + buf_size])
        stft_matrix = np.concatenate((stft_matrix, tmp_stft_matrix), 1)
    spectrogram = np.abs(stft_matrix) ** 2
    flux = get_spectral_flux(spectrogram)
    threshold = gen_local_mean_threshold(flux, local_mean_frame, 1.5)
    filtered = np.maximum(flux - threshold, 0)

    # exercise on stream
    stft_tr = STFTTransform()
    onset_tr = OnsetDetectionTransform('-p 1.5 -m {}'.format(local_mean_time).split(),
                                       **{'debug_point': 'thresholded'})

    result = np.zeros(0)
    for i in range(0, len(x), buf_size):
        tmp_stft_matrix = stft_tr.transform(x[i:i + buf_size])
        tmp_result = onset_tr.transform(tmp_stft_matrix)
        result = np.append(result, tmp_result)

    l = len(result)
    iscloses = np.isclose(result[:l], filtered[:l], rtol=1e-04)  # be more tolerant

    # verify
    assert iscloses.all(), \
        "same results between batch and transform"


@pytest.mark.parametrize('buf_size, local_mean_time, local_maximum_time', [
    [1024, 50, 20],
    [1024, 100, 20],
    [1024, 200, 20],
    [1024, 50, 200],
    [1024, 100, 200],
    [1024, 200, 200],
    [2048, 50, 20],
    [2048, 100, 20],
    [2048, 200, 20]
])
def test_generated_thresholdeds_should_be_same_with_on_batch(buf_size,
                                                             local_mean_time,
                                                             local_maximum_time):
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
    flux = get_spectral_flux(spectrogram)
    e_local_mean_threshold = gen_local_mean_threshold(flux, local_mean_frame, 1.5)
    e_local_maximum_threshold = gen_local_maximum_threshold(flux, local_maximum_frame)
    e_exponential_decay_threshold = gen_exponential_decay_threshold(flux, 0.8)

    # exercise on stream
    stft_tr = STFTTransform()
    onset_tr = OnsetDetectionTransform('-p 1.5 -m {} -M {} -d 0.8'
                                       .format(local_mean_time, local_maximum_time).split(),
                                       **{'debug_point': 'thresholds'})

    local_mean_threshold = np.zeros(0)
    local_maximum_threshold = np.zeros(0)
    exponential_decay_threshold = np.zeros(0)

    for i in range(0, len(x), buf_size):
        tmp_stft_matrix = stft_tr.transform(x[i:i + buf_size])
        tmp_results = onset_tr.transform(tmp_stft_matrix)
        local_mean_threshold = np.append(local_mean_threshold, tmp_results[0])
        local_maximum_threshold = np.append(local_maximum_threshold, tmp_results[1])
        exponential_decay_threshold = np.append(exponential_decay_threshold, tmp_results[2])

    assert (
        local_mean_threshold.shape ==
        local_maximum_threshold.shape ==
        exponential_decay_threshold.shape), \
        "each thresholds should have same form."

    l = len(local_mean_threshold)
    assert (np.isclose(local_mean_threshold, e_local_mean_threshold[:l])).all(), \
        "same results between batch and transform"
    assert (np.isclose(local_maximum_threshold, e_local_maximum_threshold[:l])).all(), \
        "same results between batch and transform"
    assert (np.isclose(exponential_decay_threshold, e_exponential_decay_threshold[:l])).all(), \
        "same results between batch and transform"
