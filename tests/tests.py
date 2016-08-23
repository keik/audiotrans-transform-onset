# -*- coding: utf-8 -*-

from audiotrans_transform_onset import OnsetDetectionTransform


def test_accept_arg_of_verbose():
    OnsetDetectionTransform(['-v'])  # no error would be raised
