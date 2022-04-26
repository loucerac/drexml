# -*- coding: utf-8 -*-
"""
Unit testing for utils module.
"""
import pytest

from dreml.utils import check_gputree_availability, get_number_cuda_devices

N_GPU_LST = [-1, 0] if check_gputree_availability() else [0]


@pytest.mark.parametrize("n_gpus_found", N_GPU_LST)
def test_get_number_cuda_devices(n_gpus_found):
    """Unit test CUDA devices found."""
    n_gpus = get_number_cuda_devices()
    assert n_gpus > 0 if n_gpus_found > 0 else n_gpus == 0
