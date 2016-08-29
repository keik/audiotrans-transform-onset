# -*- coding: utf-8 -*-

from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter
from logging import getLogger, StreamHandler, Formatter, DEBUG
from audiotrans import Transform

logger = getLogger(__package__)
handler = StreamHandler()
handler.setFormatter(Formatter('[%(asctime)s %(levelname)s %(name)s] %(message)s'))
logger.addHandler(handler)


class OnsetDetectionTransform(Transform):

    def __init__(self, argv=[]):
        parser = ArgumentParser(
            prog='onset',
            description="""audiotrans transform module for onset detection.

Transform from STFT matrix to list of times of onset,
or spectral flux with each thresholds for debug.

Returned times of onset is in millisecond of onset detected position.""",
            formatter_class=RawTextHelpFormatter)

        parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                            help='Run as verbose mode')

        parser.add_argument('-r', '--framerate', dest='framerate', default='44100',
                            help="Framerate of original wave. Default is 44100")

        parser.add_argument('-H', '--hop-size', dest='hop_size', default='256',
                            help="Hop size of STFT on ipnut STFT matrix")

        parser.add_argument('-m', '--local-mean-time', dest='local_mean_time', default='150',
                            help="Threshold value of feature to detect onset")

        parser.add_argument('-p', '--local-mean-multiplier', dest='local_mean_multiplier',
                            default='1.5',
                            help="Multiplier to local means to generate threshold function")

        parser.add_argument('-M', '--local-maximum-time', dest='local_maximum_time', default='60',
                            help="Threshold value of feature to detect onset")

        parser.add_argument('-d', '--decay-factor', dest='decay_factor', default='0.8',
                            help="Threshold value of feature to detect onset")

        parser.add_argument('-D', '--debug', dest='debug', action='store_true',
                            help="Transform to spectral flux and each thresholds "
                            "instead of times of onsets")

        args = parser.parse_args(argv)

        if args.verbose:
            logger.setLevel(DEBUG)
            logger.info('Start as verbose mode')

        self.framerate = int(args.framerate)
        self.hop_size = int(args.hop_size)
        self.ms_per_frame = self.hop_size / self.framerate * 1000
        self.mean_frame_size = int(int(args.local_mean_time) / self.ms_per_frame)
        self.maximum_frame_size = int(int(args.local_maximum_time) / self.ms_per_frame)
        half = (max(self.mean_frame_size, self.maximum_frame_size) - 1) / 2
        self.back_cache_size = int(np.ceil(half))
        self.next_cache_size = int(np.floor(half))
        self.local_mean_multiplier = float(args.local_mean_multiplier)
        self.decay_factor = float(args.decay_factor)

        self.old_spectrogram = None
        self.old_spectral_flux = None
        self.last_exponential_decay_threshold = np.empty(0)
        self.window_size = None
        self.total_frame_count = 0

        self.debug = args.debug

    def transform(self, stft_matrix):
        if self.window_size is None:
            self.window_size = (stft_matrix.shape[0] - 1) * 2

        # get power spectrogram from STFT matrix
        spectrogram = np.abs(stft_matrix) ** 2

        # merge old power spectrogram to calculate difference between full neighbored frames
        if self.old_spectrogram is None:
            self.old_spectrogram = np.zeros((spectrogram.shape[0], 0))
        merged_spectrogram = np.concatenate((self.old_spectrogram, spectrogram), 1)
        self.old_spectrogram = merged_spectrogram[:, -1:]

        # calculate spectral flux from power spectrogram
        tmp_spectral_flux = get_spectral_flux(merged_spectrogram)
        logger.info('transform from {}-spectrogram to temporary {}-spectral flux'
                    .format(merged_spectrogram.shape, tmp_spectral_flux.shape))

        # merge old spectral flux
        if self.old_spectral_flux is None:
            self.old_spectral_flux = np.zeros(0)
        merged_spectral_flux = np.concatenate((self.old_spectral_flux, tmp_spectral_flux), 0)
        self.old_spectral_flux = merged_spectral_flux[-(self.next_cache_size +
                                                        self.back_cache_size):]

        # detect range of slice to be able to reconstruct flux and each thresholds
        s = -(self.next_cache_size + len(tmp_spectral_flux))
        e = -(self.next_cache_size)

        # threshold 1: local mean
        local_mean_threshold = gen_local_mean_threshold(
            merged_spectral_flux,
            self.mean_frame_size,
            self.local_mean_multiplier)[s:e]

        # threshold 2: local maximum
        local_maximum_threshold = gen_local_maximum_threshold(
            merged_spectral_flux,
            self.maximum_frame_size)[s:e]

        # threshold 3: exponential decay
        exponential_decay_threshold = gen_exponential_decay_threshold(
            merged_spectral_flux[s:e],
            self.decay_factor,
            self.last_exponential_decay_threshold)
        self.last_exponential_decay_threshold = exponential_decay_threshold[-1:]

        # slice spectral flux and apply each threshold functions
        spectral_flux = merged_spectral_flux[s:e]

        if self.debug:
            return (spectral_flux,
                    local_mean_threshold,
                    local_maximum_threshold,
                    exponential_decay_threshold)

        is_onsets = ((spectral_flux - local_mean_threshold >= 0) *
                     (spectral_flux - local_maximum_threshold >= 0) *
                     (spectral_flux - exponential_decay_threshold >= 0))

        # calculate times of onsets occured
        onsets = (np.where(is_onsets)[0] + self.total_frame_count) * self.ms_per_frame

        if len(onsets) > 0:
            logger.info('onsets were detected at [{}] (ms)'.format(onsets))

        self.total_frame_count += spectral_flux.shape[0]

        return onsets


def get_spectral_flux(spectrogram):
    """
    Returns positive spectral flux from spectrogram
    """

    p_incremental_diff = np.maximum(spectrogram[:, 1:] - spectrogram[:, :-1], 0)
    p_diff_means = np.mean(p_incremental_diff, 0)

    return p_diff_means


def gen_local_mean_threshold(x, local_frame_size, multiplier):
    """
    Returns local mean of specified array data,
    within specified range,
    with multipled by specifiedmultiplier
    """

    return uniform_filter(x, local_frame_size) * multiplier


def gen_local_maximum_threshold(x, local_frame_size):
    """
    Returns local maximum of specified array data,
    within specified range
    """

    return maximum_filter(x, local_frame_size)


def gen_exponential_decay_threshold(x, decay_fact, acc=np.zeros(1)):
    """
    Returns array of exponential decay threshold.
    If current value are greater than decayed value, then
    the value are update with current value.
    """

    if len(acc) == 0:
        acc = np.zeros(1)
    if len(x) == 0:
        return acc[1:]
    return gen_exponential_decay_threshold(
        x[1:],
        decay_fact,
        np.append(acc, max(x[0], decay_fact * acc[-1] + (1 - decay_fact) * x[0])))
