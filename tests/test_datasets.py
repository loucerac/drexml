# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Unit testing for datasets module.
"""
import importlib.resources as pkg_resources
import shutil
import tempfile
from pathlib import Path

import click
import joblib
import pytest
from click.testing import CliRunner

from dreml.cli.cli import main
from dreml.datasets import get_disease_data, load_df

DATA_NAMES = [
    "circuits.tsv.gz",
    "gene_exp.tsv.gz",
    "pathvals.tsv.gz",
    "genes.tsv.gz",
]


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


@pytest.mark.parametrize("fname", DATA_NAMES)
def test_load_df(fname):
    """Test load_df"""

    fpath = get_resource_path(fname)
    data = load_df(fpath)
    assert data.shape[0] > 0


def make_disease_path(default=False):
    """Prepare fake disease data folder."""
    tmp_dir = Path(tempfile.mkdtemp())
    if default:
        disease_path_in = get_resource_path("experiment_default.env")
    else:
        disease_path_in = get_resource_path("experiment.env")
    disease_path_out = tmp_dir.joinpath(disease_path_in.name)
    # Use as_posix to make it compatible with python<=3.7
    shutil.copy(disease_path_in.as_posix(), disease_path_out.as_posix())

    with open(disease_path_out, "r", encoding="utf8") as this_file:
        disease_path_out_data = this_file.read()
    if not default:
        disease_path_out_data = disease_path_out_data.replace(
            "%THIS_PATH", disease_path_in.parent.as_posix()
        )
    with open(disease_path_out, "w", encoding="utf8") as this_file:
        this_file.write(disease_path_out_data)

    if default:
        shutil.copy(
            get_resource_path("circuits.tsv.gz").as_posix(),
            disease_path_out.parent.as_posix(),
        )

    return disease_path_out


def test_get_disease_data():
    """Test get_disease_data."""

    disease_path = make_disease_path(default=False)
    gene_exp, pathvals, circuits, genes = get_disease_data(disease_path, debug=True)

    assert gene_exp.to_numpy().ndim == 2
    assert pathvals.to_numpy().ndim == 2
    assert circuits.to_numpy().ndim == 2
    assert genes.to_numpy().ndim == 2


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize("default", [True, False])
def test_orchestrate(debug, default):
    """Unit tests for CLI app."""
    click.echo("Running CLI tests fro DREML.")

    disease_path = make_disease_path(default)

    opts = ["orchestrate", "--debug" if debug else "--no-debug", f"{disease_path}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(main, " ".join(opts))

    ml_folder_expected = disease_path.parent.joinpath("ml")
    tmp_folder_expected = ml_folder_expected.joinpath("tmp")

    fpath = tmp_folder_expected.joinpath("features.jbl")
    features = joblib.load(tmp_folder_expected.joinpath("features.jbl"))
    if debug:
        assert (fpath.exists()) and (features.shape[0] == 9)
    else:
        assert (fpath.exists()) and (features.shape[0] > 9)


# @pytest.mark.parametrize("n_gpus", [0, -1])
# def test_cli_single(n_gpus):
#     """Unit tests for CLI app."""
#     click.echo("Running CLI tests fro DREML.")

#     disease_path = make_disease_path(default=False)

#     ml_folder_expected = disease_path.parent.joinpath("ml")
#     tmp_folder_expected = ml_folder_expected.joinpath("tmp")

#     opts = ["orchestrate", "--debug", f"{disease_path}"]
#     click.echo(" ".join(opts))
#     runner = CliRunner()
#     runner.invoke(main, " ".join(opts))

#     opts = [
#         "stability",
#         "--mode train",
#         "--debug",
#         f"--n-gpus {n_gpus}",
#         f"{disease_path}",
#     ]
#     click.echo(" ".join(opts))
#     runner = CliRunner()
#     runner.invoke(main, " ".join(opts))

#     model_fpath = tmp_folder_expected.joinpath("model_0.jbl")
#     assert model_fpath.exists()

#     opts = [
#         "stability",
#         "--mode explain",
#         "--debug",
#         f"--n-gpus {n_gpus}",
#         f"{disease_path}",
#     ]
#     click.echo(" ".join(opts))
#     runner = CliRunner()
#     runner.invoke(main, " ".join(opts))

#     model_fpath = tmp_folder_expected.joinpath("fs.jbl")
#     assert model_fpath.exists()

#     opts = [
#         "stability",
#         "--mode score",
#         "--debug",
#         f"--n-gpus {n_gpus}",
#         f"{disease_path}",
#     ]
#     click.echo(" ".join(opts))
#     runner = CliRunner()
#     runner.invoke(main, " ".join(opts))

#     stability_results_fpath = ml_folder_expected.joinpath("stability_results.tsv")
#     assert stability_results_fpath.exists()

#     opts = [
#         "explain",
#         "--debug",
#         f"--n-gpus {n_gpus}",
#         f"{disease_path}",
#     ]
#     click.echo(" ".join(opts))
#     runner = CliRunner()
#     runner.invoke(main, " ".join(opts))

#     shap_selection_fpath = ml_folder_expected.joinpath("shap_selection.tsv")
#     shap_summary_fpath = ml_folder_expected.joinpath("shap_summary.tsv")
#     assert shap_selection_fpath.exists() and shap_summary_fpath.exists()


@pytest.mark.parametrize("n_gpus", [0, -1])
def test_cli_run(n_gpus):
    """Unit tests for CLI app."""
    click.echo("Running CLI tests fro DREML.")

    disease_path = make_disease_path(default=False)
    ml_folder_expected = disease_path.parent.joinpath("ml")

    opts = ["run", "--debug", f"--n-gpus {n_gpus}", f"{disease_path.as_posix()}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(main, " ".join(opts))

    exist_files = [
        ml_folder_expected.joinpath(fname).exists()
        for fname in ["stability_results.tsv", "shap_selection.tsv", "shap_summary.tsv"]
    ]

    assert all(exist_files)
