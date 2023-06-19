# -*- coding: utf-8 -*-
"""
Unit testing for utils module.
"""
import pathlib
from tempfile import mkstemp

import click
import numpy as np
import pandas as pd
import pytest
import shap
from click.testing import CliRunner

from drexml.cli.cli import main
from drexml.config import DEFAULT_DICT
from drexml.utils import (
    check_cli_arg_is_bool,
    check_gputree_availability,
    convert_names,
    get_number_cuda_devices,
    get_out_path,
    get_resource_path,
    get_stab,
    get_version,
    parse_stab,
    read_disease_config,
)

from .this_utils import make_disease_config

RESOURCE_FNAMES = [
    "circuit_names.tsv",
    "circuits2genes_gtex-v8_hipathia-v2-14-0.tsv.gz",
    "genes_drugbank-v050110_gtex-v8_mygene-v20230220.tsv.gz",
]


@pytest.mark.parametrize("fname", RESOURCE_FNAMES)
def test_get_resource_path(fname):
    """Test get_resource_path"""

    fpath = get_resource_path(fname)
    assert fpath.exists()


def test_get_number_cuda_devices():
    """Unit test CUDA devices found if GPUTreeExplainer is avilable."""
    n_gpus = get_number_cuda_devices()
    if check_gputree_availability():
        assert n_gpus > 0


def test_check_gputree_availability():
    """Unit test CUDA devices found if GPUTreeExplainer is avilable."""

    is_available = check_gputree_availability()

    assert isinstance(is_available, bool)


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


@pytest.mark.xfail(raises=NotImplementedError)
def test_get_out_path_notok():
    """
    Unit test out path from bad config file.
    """
    path = mkstemp(suffix=".txt", prefix="abc")[1]
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


@pytest.mark.parametrize("debug_str", ["1", "0", "True", "False", "true", "false"])
@pytest.mark.parametrize("mode_str", ["final", "stab"])
def test_parse_stab_ok(debug_str, mode_str):
    """
    Unit test that parse_stab returns the correct number of arguments when feeded with a
    sys-like argv.
    """

    args = ["file.py", "./test", "0", "4", "32", debug_str, "1", mode_str]
    parsed_args = parse_stab(args)

    assert len(parsed_args) == 7
    data_folder, n_iters, n_gpus, n_cpus, n_splits, debug, add = parsed_args

    assert isinstance(data_folder, (str, pathlib.PurePath))
    assert isinstance(n_iters, int)
    assert isinstance(n_gpus, int)
    assert isinstance(n_cpus, int)
    assert isinstance(n_splits, int)

    assert isinstance(debug, bool)

    if debug_str in ["1", "True", "true"]:
        assert debug is True
    else:
        assert debug is False

    if mode_str in ["final"]:
        assert n_splits == 1
    else:
        if debug:
            assert n_splits == 3

        else:
            assert n_splits == 100

    assert isinstance(add, bool)


@pytest.mark.parametrize("arg", ["1", "True", "true"])
def test_check_cli_arg_is_bool_istrue(arg):
    """
    Unit test that check_cli_arg_is_bool returns True when fed with a sys-like argv.
    """

    assert check_cli_arg_is_bool(arg) is True


@pytest.mark.parametrize("arg", ["0", "False", "false"])
def test_check_cli_arg_is_bool_isfalse(arg):
    """
    Unit test that check_cli_arg_is_bool returns False when fed with a sys-like argv.
    """

    assert check_cli_arg_is_bool(arg) is False


def test_check_cli_arg_is_bool_isnotok():
    """
    Unit test that check_cli_arg_is_bool returns False when fed with a sys-like argv.
    """

    with pytest.raises(Exception):
        check_cli_arg_is_bool(10)


def test_get_stab():
    """
    Unit test that get_stab returns the expected features and splits.
    """

    click.echo("Running CLI tests for DREXML.")

    disease_path = make_disease_config(use_seeds=True, update=False)
    data_folder = disease_path.parent.joinpath("results", "tmp")

    opts = ["orchestrate", f"{disease_path.as_posix()}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(main, " ".join(opts))

    # subprocess.run(["drexml"] + opts)

    n_splits = 100
    n_cpus = 1
    debug = False
    n_iters = 0

    n_features = 698
    n_targets = 2

    model, stab_cv, features_orig, targets_orig = get_stab(
        data_folder, n_splits, n_cpus, debug, n_iters
    )

    assert shap.utils.safe_isinstance(model, "sklearn.ensemble.RandomForestRegressor")
    assert model.n_estimators == 200
    assert model.max_depth == 8
    assert model.max_features == int(np.sqrt(n_features) + 20)
    assert len(stab_cv) == n_splits
    assert all([len(i) == 3 for i in stab_cv])
    assert features_orig.shape[1] == n_features
    assert targets_orig.shape[1] == n_targets


def test_convert_names_genes():
    """
    Unit test that convert_names returns the expected features and targets.
    """
    data = pd.DataFrame(np.random.rand(3, 1), columns=["2175"])

    data_out = convert_names(data, keys=("genes",), axis=(1,))
    assert data_out.shape == (3, 1)
    assert data_out.columns[0] == "FANCA"


def test_convert_names_circuits():
    """
    Unit test that convert_names returns the expected features and targets.
    """
    data = pd.DataFrame(np.random.rand(3, 1), columns=["P.hsa03320.28"])

    data_out = convert_names(data, keys=("circuits",), axis=(1,))
    assert data_out.shape == (3, 1)
    assert data_out.columns[0] == "PPAR signaling pathway: ACSL1"


@pytest.mark.xfail(raises=NotImplementedError)
def test_convert_names_keynotok():
    """Unit test that convert_names raises an error."""
    data = pd.DataFrame(np.random.rand(3, 1), columns=["P.hsa03320.28"])

    convert_names(data, keys=("vader",), axis=(1,))


@pytest.mark.xfail(raises=ValueError)
def test_convert_names_axisnotok():
    """Unit test that convert_names raises an error."""
    data = pd.DataFrame(np.random.rand(3, 1), columns=["P.hsa03320.28"])

    convert_names(data, keys=("circuits",), axis=(2,))
