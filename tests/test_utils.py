# -*- coding: utf-8 -*-
"""
Unit testing for utils module.
"""
from dreml.utils import get_number_cuda_devices


def test_get_number_cuda_devices():
    """Unit test CUDA evices found."""
    n_gpus = get_number_cuda_devices()
    assert n_gpus > 0
