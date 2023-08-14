# -*- coding: utf-8 -*-
"""
Unit testing for datasets module.
"""

import click
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from drexml.cli.cli import main
from drexml.utils import check_gputree_availability

from .this_utils import THIS_DIR, make_disease_config

PLOTTING_EXTENSIONS = ["pdf", "png"]
N_GPU_LST = [-1, 0] if check_gputree_availability() else [0]


def check_new_versus_all(name, tmp_fodler):
    old_df = pd.read_csv(THIS_DIR.joinpath(name), sep="\t", index_col=0)
    new_df = pd.read_csv(tmp_fodler.joinpath(name), sep="\t", index_col=0)

    return np.allclose(old_df, new_df, equal_nan=True)


@pytest.mark.parametrize("n_gpus", N_GPU_LST)
def test_cli_run(n_gpus):
    """Unit tests for CLI app."""

    click.echo("Running CLI tests for DREXML.")

    disease_path = make_disease_config(use_seeds=True, update=False)
    ml_folder_expected = disease_path.parent.joinpath("results")

    opts = [
        "run",
        "--verbosity",
        "--debug",
        f"--n-gpus {n_gpus}",
        f"{disease_path.as_posix()}",
    ]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(main, " ".join(opts))

    expected_files = ["stability_results.tsv", "shap_selection.tsv", "shap_summary.tsv"]

    exist_files = [ml_folder_expected.joinpath(fname) for fname in expected_files]

    assert all([x.exists() for x in exist_files])

    numeric_checks = [
        check_new_versus_all(fname, ml_folder_expected) for fname in expected_files
    ]
    assert all(numeric_checks)

    renamed_files = [
        ml_folder_expected.joinpath(f"{x.stem}_symbol.tsv") for x in exist_files
    ]

    opts = ["rename", f"{ml_folder_expected.as_posix()}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(main, " ".join(opts))

    assert all([x.exists() for x in renamed_files])

    plot_files = [
        ml_folder_expected.joinpath(f"{name}.{ext}")
        for ext in PLOTTING_EXTENSIONS
        for name in ["metrics"]
    ]

    opts = [
        "plot",
        ml_folder_expected.joinpath("shap_selection_symbol.tsv").as_posix(),
        ml_folder_expected.joinpath("shap_summary_symbol.tsv").as_posix(),
        ml_folder_expected.joinpath("stability_results_symbol.tsv").as_posix(),
        ml_folder_expected.as_posix(),
    ]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(main, " ".join(opts))

    assert all([x.exists() for x in plot_files])


@pytest.mark.xfail()
def test_cli_run_step_fails():
    """Test that cli fails when skipping an step."""

    n_gpus = 0
    click.echo("Running CLI tests for DREXML.")

    disease_path = make_disease_config(use_seeds=True, update=False)

    runner = CliRunner()

    orchestrate_opts = [
        "orchestrate",
        "--debug",
        f"--n-gpus {n_gpus}",
        f"{disease_path.as_posix()}",
    ]
    click.echo(" ".join(orchestrate_opts))
    runner.invoke(main, " ".join(orchestrate_opts))


@pytest.mark.xfail()
def test_cli_run_mode_fails():
    """Test that cli fails when skipping an step."""

    n_gpus = 0
    click.echo("Running CLI tests for DREXML.")

    disease_path = make_disease_config(use_seeds=True, update=False)

    runner = CliRunner()

    orchestrate_opts = [
        "orchestrate",
        "--debug",
        f"--n-gpus {n_gpus}",
        f"{disease_path.as_posix()}",
    ]
    click.echo(" ".join(orchestrate_opts))
    runner.invoke(main, " ".join(orchestrate_opts))

    train_opts = [
        "stability --mode train",
        "--debug",
        f"--n-gpus {n_gpus}",
        f"{disease_path.as_posix()}",
    ]
    runner.invoke(main, " ".join(train_opts))

    cmd_opts = [
        "stability --mode vader",
        "--debug",
        f"--n-gpus {n_gpus}",
        f"{disease_path.as_posix()}",
    ]
    runner.invoke(main, " ".join(cmd_opts))
