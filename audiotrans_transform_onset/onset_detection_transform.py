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

    def __init__(self, argv=[], **kwargs):
        parser = ArgumentParser(
            prog='onset',
            description="""audiotrans transform module for onset detection.

Transform from STFT matrix to list of onsets.
One onset is constructed 2-D vector like `[time, feature]`.

`time` is millisecond of onset detected position.
`feature` is value of feature which be able to specify by `--feature` option.""",
            formatter_class=RawTextHelpFormatter)

        parser.add_argument('-v', '--verbose', dest='verbose',
                            action='store_true',
                            help='Run as verbose mode')

        parser.add_argument('-r', '--framerate', dest='framerate', default='44100',
                            help="Framerate of original wave. Default is 44100")

        parser.add_argument('-H', '--hop-size', dest='hop_size', default='256',
                            help="Hop size of STFT on ipnut STFT matrix")

        parser.add_argument('-m', '--local-mean-time', dest='local_mean_time', default='200',
                            help="Threshold value of feature to detect onset")

        parser.add_argument('-M', '--local-maximum-time', dest='local_maximum_time', default='200',
                            help="Threshold value of feature to detect onset")

        parser.add_argument('-d', '--decay_factor', dest='decay_factor', default='0.8',
                            help="Threshold value of feature to detect onset")

        parser.add_argument('-p', '--local-mean-multiplier', dest='local_mean_multiplier',
                            default='1.5',
                            help="Multiplier to local means to generate threshold function")

        parser.add_argument('-F', '--feature-threshold', dest='feature_threshold', default='1000',
                            help="Threshold value of feature to detect onset")

        parser.add_argument('-T', '--time-threshold', dest='time_threshold', default='100',
                            help="Threshold time to fold neigbored frames to detect onset")

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
        self.feature_threshold = float(args.feature_threshold)
        self.time_threshold = float(args.time_threshold)

        self.old_spectrogram = None
        self.old_spectral_flux = None
        self.last_exponential_decay_threshold = np.empty(0)
        self.window_size = None
        self.total_frame_count = 0
        self.onset_seq = np.zeros(0).reshape(-1, 2)

        self.__debug_point = kwargs.pop('debug_point', None)

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

        if self.__debug_point == 'flux':
            return tmp_spectral_flux

        # merge old spectral flux
        if self.old_spectral_flux is None:
            self.old_spectral_flux = np.zeros(0)
        merged_spectral_flux = np.concatenate((self.old_spectral_flux, tmp_spectral_flux), 0)
        self.old_spectral_flux = merged_spectral_flux[-(self.next_cache_size +
                                                        self.back_cache_size):]

        # calcurate threshold from means of spectral flux
        s = -(self.next_cache_size + len(tmp_spectral_flux))
        e = -(self.next_cache_size)

        local_mean_threshold = gen_local_mean_threshold(
            merged_spectral_flux,
            self.mean_frame_size,
            self.local_mean_multiplier)[s:e]

        local_maximum_threshold = gen_local_maximum_threshold(
            merged_spectral_flux,
            self.maximum_frame_size)[s:e]

        exponential_decay_threshold = gen_exponential_decay_threshold(
            merged_spectral_flux[s:e],
            self.decay_factor,
            self.last_exponential_decay_threshold)
        self.last_exponential_decay_threshold = exponential_decay_threshold[-1:]

        spectral_flux = merged_spectral_flux[s:e]
        filtered_flux = np.maximum(spectral_flux - local_mean_threshold, 0)

        logger.info(('thresholded merged {}-spectral flux with local mean of ' +
                    '{}-frames to {}-spectral flux')
                    .format(merged_spectral_flux.shape, self.mean_frame_size, filtered_flux.shape))

        if self.__debug_point == 'thresholds':
            return local_mean_threshold, local_maximum_threshold, exponential_decay_threshold

        if self.__debug_point == 'thresholded':
            return filtered_flux

        # TODO: pick peak
        return spectral_flux, local_mean_threshold, local_maximum_threshold


def get_spectral_flux(spectrogram):
    """
    get incremental power difference between previous frames
    and fold power difference by getting mean of each frequency bins
    """
    p_incremental_diff = np.maximum(spectrogram[:, 1:] - spectrogram[:, :-1], 0)
    p_diff_means = np.mean(p_incremental_diff, 0)

    return p_diff_means


def gen_local_mean_threshold(flux, local_frame_size, multiplier):
    return uniform_filter(flux, local_frame_size) * multiplier


def gen_local_maximum_threshold(flux, local_frame_size):
    return maximum_filter(flux, local_frame_size)


def gen_exponential_decay_threshold(x, decay_fact, acc=np.zeros(1)):
    if len(acc) == 0:
        acc = np.zeros(1)
    if len(x) == 0:
        return acc[1:]
    return gen_exponential_decay_threshold(
        x[1:],
        decay_fact,
        np.append(acc, max(x[0], decay_fact * acc[-1] + (1 - decay_fact) * x[0])))
