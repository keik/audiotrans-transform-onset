# -*- coding: utf-8 -*-

from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
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

        parser.add_argument('-m', '--mean-frame_size', dest='mean_frame_size', default='30',
                            help="Threshold value of feature to detect onset")

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
        self.mean_frame_size = int(args.mean_frame_size)
        self.mean_frame_prev_size = int(np.ceil((self.mean_frame_size - 1) / 2))
        self.mean_frame_next_size = int(np.floor((self.mean_frame_size - 1) / 2))
        self.feature_threshold = float(args.feature_threshold)
        self.time_threshold = float(args.time_threshold)

        self.old_spectrogram = None
        self.old_spectral_flux = None
        self.window_size = None
        self.total_frame_count = 0
        self.onset_seq = np.empty(0).reshape(-1, 2)

    def transform(self, stft_matrix):
        if self.window_size is None:
            self.window_size = (stft_matrix.shape[0] - 1) * 2
            self.ms_per_frame = self.window_size / self.framerate * 1000 / stft_matrix.shape[1]

        # get power spectrogram from STFT matrix
        spectrogram = np.abs(stft_matrix) ** 2

        # merge old power spectrogram to calculate difference between full neighbored frames
        if self.old_spectrogram is None:
            self.old_spectrogram = np.empty((spectrogram.shape[0], 0))
        merged_spectrogram = np.concatenate((self.old_spectrogram, spectrogram), 1)
        self.old_spectrogram = merged_spectrogram[:, -1:]

        # calculate spectral flux from power spectrogram
        tmp_spectral_flux = self._spectral_flux(merged_spectrogram)

        # merge old spectral flux
        if self.old_spectral_flux is None:
            self.old_spectral_flux = np.empty(0)
        merged_spectral_flux = np.concatenate((self.old_spectral_flux, tmp_spectral_flux), 0)
        self.old_spectral_flux = merged_spectral_flux[-self.mean_frame_size + 1:]

        # calcurate threshold from means of spectral flux
        threshold = self._get_local_means(merged_spectral_flux, self.mean_frame_size, 1.5)

        # thresholding spectral flux by means
        spectral_flux = merged_spectral_flux[self.mean_frame_prev_size:
                                             self.mean_frame_prev_size + len(threshold)]
        filtered_flux = np.maximum(spectral_flux - threshold, 0)

        self.total_frame_count += stft_matrix.shape[1]

        # TODO: pick peak
        return filtered_flux, threshold

    def _spectral_flux(self, spectrogram):

        # get incremental power difference between previous frames
        # and fold power difference by getting mean of each frequency bins
        # TODO: add another flux like negative flux and difference flux
        # TODO: parametrize norm order
        p_incremental_diff = np.maximum(spectrogram[:, 1:] - spectrogram[:, :-1], 0)
        p_diff_means = np.mean(p_incremental_diff, 0)

        logger.info('calculated {}-spectral flux from {}-spectrogram'
                    .format(p_diff_means.shape, spectrogram.shape))

        return p_diff_means

    def _get_local_means(self, spectral_flux, mean_frame_size, multiplier):
        threshold = (np.array([np.mean(spectral_flux[i:self.mean_frame_size + i])
                               for i in range(len(spectral_flux) - mean_frame_size + 1)]) *
                     multiplier)

        logger.info('calculated {}-mean of spectral flux in range of {} from {}-spectral flux'
                    .format(threshold.shape, mean_frame_size, spectral_flux.shape))

        return threshold
