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

        parser.add_argument('-f', '--feature', dest='feature', default='diff',
                            help="""To use as feature value. Default is `diff`.
Below values are available:

  abs  : Power of spectrum for each frames
  diff : Difference of power of spectrum for each neighored frames

""")

        parser.add_argument('-T', '--time-threshold', dest='time_threshold', default='1000',
                            help="Threshold time to fold neigbored frames to detect onset")

        parser.add_argument('-F', '--feature-threshold', dest='feature_threshold', default='1000',
                            help="Threshold value of feature to detect onset")

        args = parser.parse_args(argv)

        if args.verbose:
            logger.setLevel(DEBUG)
            logger.info('Start as verbose mode')

        self.hop_size = int(args.hop_size)
        self.prev_wave = np.empty(0)

    def transform(self, wave):

        # TODO: implement
        onsets = np.empty(0)
        return onsets
