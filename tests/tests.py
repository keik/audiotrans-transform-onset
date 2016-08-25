# -*- coding: utf-8 -*-

import pytest
import wave
import numpy as np
from scipy.ndimage import uniform_filter
from audiotrans_transform_stft import STFTTransform
from audiotrans_transform_onset import OnsetDetectionTransform
from audiotrans_transform_onset import get_spectral_flux


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
    multiplier = 1.5
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
    threshold = uniform_filter(flux, size=local_mean_frame) * multiplier
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
