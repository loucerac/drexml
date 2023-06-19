# -*- coding: utf-8 -*-
"""
Unit testing for utils module.
"""
from tempfile import mkstemp

import pytest

from drexml.config import DEFAULT_DICT
from drexml.utils import (
    check_gputree_availability,
    get_number_cuda_devices,
    get_out_path,
    get_resource_path,
    get_version,
    read_disease_config,
)

from .this_utils import make_disease_config

N_GPU_LST = [-1, 0] if check_gputree_availability() else [0]
WITH_GPU = [True] if get_number_cuda_devices() > 0 else [False]


RESOURCE_FNAMES = [
    "circuit_names.tsv",
    "circuits.tsv.gz",
    "gene_exp.tsv.gz",
    "pathvals.tsv.gz",
    "circuits2genes_gtex-v8_hipathia-v2-14-0.tsv.gz",
    "genes_drugbank-v050110_gtex-v8_mygene-v20230220.tsv.gz",
]


@pytest.mark.parametrize("fname", RESOURCE_FNAMES)
def test_get_resource_path(fname):
    """Test get_resource_path"""

    fpath = get_resource_path(fname)
    assert fpath.exists()


@pytest.mark.parametrize("n_gpus_found", N_GPU_LST)
def test_get_number_cuda_devices(n_gpus_found):
    """Unit test CUDA devices found if GPUTreeExplainer is avilable."""
    n_gpus = get_number_cuda_devices()
    assert n_gpus > 0 if n_gpus_found > 0 else True


@pytest.mark.parametrize("gpu_available", WITH_GPU)
def test_check_gputree_availability(gpu_available):
    """Unit test CUDA devices found if GPUTreeExplainer is avilable."""
    is_available = check_gputree_availability()

    assert is_available == gpu_available


def test_get_version():
    """
    Unit test version string.
    """

    ver = get_version()
    assert ver is not None
    assert isinstance(ver, str)
    assert len(ver) > 0
    assert ver[0].isnumeric()
    assert len(ver.split(".")) == 3


def test_get_out_path_ok():
    """
    Unit test out path.
    """

    path = make_disease_config(use_seeds=True)

    assert get_out_path(path).exists() is True


def test_get_out_path_notok():
    """
    Unit test out path from bad config file.
    """
    path = mkstemp(suffix=".txt", prefix="abc")
    with pytest.raises(Exception):
        get_out_path(path)


def test_read_disease_config_only_seeds():
    """
    Unit test read disease config.
    """

    path = make_disease_config(use_seeds=True, update=False)
    print(path)

    config = read_disease_config(path)

    assert len(config["seed_genes"]) == 1
    assert config["seed_genes"][0] == "2180"


def test_read_disease_config_update_with_seeds():
    """
    Unit test read disease config.
    """

    path = make_disease_config(use_seeds=True, update=True)
    print(path)

    config = read_disease_config(path)

    assert len(config["seed_genes"]) == 1
    assert config["seed_genes"][0] == "2180"
    assert config["pathvals"] != DEFAULT_DICT["pathvals"]
    assert config["genes"] != DEFAULT_DICT["genes"]
    assert config["circuits"].exists()
    assert config["gene_exp"] != DEFAULT_DICT["gene_exp"]


def test_read_disease_config_update_without_seeds():
    """
    Unit test read disease config.
    """

    path = make_disease_config(use_seeds=False, update=True)
    print(path)

    config = read_disease_config(path)

    assert config["seed_genes"] is None
    assert config["pathvals"] != DEFAULT_DICT["pathvals"]
    assert config["genes"] != DEFAULT_DICT["genes"]
    assert config["circuits"] != DEFAULT_DICT["circuits"]
    assert config["gene_exp"] != DEFAULT_DICT["gene_exp"]
