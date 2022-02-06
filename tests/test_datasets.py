# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Unit testing for datasets module.
"""
import shutil
import tempfile
from pathlib import Path
import joblib

import pytest

import dreml.cli.orchestrate

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import click
from click.testing import CliRunner

from dreml.cli import orchestrate
from dreml.datasets import get_disease_data


def get_resource_path(fname):
    """Get path to example disease env path.

    Returns
    -------
    pathlib.PosixPath
        Path to file.
    """
    with pkg_resources.path("dreml.resources", fname) as f:
        data_file_path = f
    return Path(data_file_path)


@pytest.fixture
def disease_path():
    """Prepare fake disease data folder."""
    tmp_dir = Path(tempfile.mkdtemp())
    disease_path_in = get_resource_path("experiment.env")
    disease_path_out = tmp_dir.joinpath(disease_path_in.name)
    # Use as_posix to make it compatible with python<=3.7
    shutil.copy(disease_path_in.as_posix(), disease_path_out.as_posix())

    with open(disease_path_out, "r", encoding="utf8") as this_file:
        disease_path_out_data = this_file.read()
    disease_path_out_data = disease_path_out_data.replace(
        "%THIS_PATH", disease_path_in.parent.as_posix()
    )
    with open(disease_path_out, "w", encoding="utf8") as this_file:
        this_file.write(disease_path_out_data)

    return disease_path_out


def test_get_disease_data(disease_path):
    """Test get_disease_data."""

    gene_exp, pathvals, circuits, genes = get_disease_data(disease_path)

    assert gene_exp.to_numpy().ndim == 2
    assert pathvals.to_numpy().ndim == 2
    assert circuits.to_numpy().ndim == 2
    assert genes.to_numpy().ndim == 2


@pytest.mark.parametrize("debug", [(True,), (False,)])
def test_orchestrate(disease_path, debug):
    """Unit tests for CLI app."""
    click.echo("Running CLI tests fro DREML.")

    opts = ["--debug" if debug else "--no-debug", f"{disease_path}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(dreml.cli.orchestrate.orchestrate, " ".join(opts))

    ml_name = "ml" if debug else "debug"
    ml_folder_expected = disease_path.parent.joinpath(ml_name)
    tmp_folder_expected = ml_folder_expected.joinpath("tmp")

    assert ml_folder_expected.exists()
    assert tmp_folder_expected.exists()

    features = joblib.load(tmp_folder_expected.joinpath("features.jbl"))
    if debug:
        assert features.shape[0] == 9
    else:
        assert features.shape[0] > 9
