# -*- coding: utf-8 -*-
"""
Unit testing for utils module.
"""
import pathlib
import tempfile
from tempfile import mkstemp

import click
import numpy as np
import pandas as pd
import pytest
import shap
import utils
from click.testing import CliRunner

from drexml.plotting import plot_metrics
from drexml.cli.cli import main
from drexml.config import DEFAULT_DICT, VERSION_DICT
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

from .this_utils import make_disease_config, THIS_DIR

PLOTTING_EXTENSIONS = ["pdf", "png"]

def test_plot_metrics():
    """
    Test plotting metrics.
    """
    fpath = pathlib.Path(THIS_DIR, "stability_results.tsv")
    tmp_folder = pathlib.Path(tempfile.mkdtemp())

    plot_files = [
        tmp_folder.joinpath(f"{x.stem}.{ext}")
        for ext in PLOTTING_EXTENSIONS
        for x in [fpath]
    ]
     
    plot_metrics(input_path=fpath, output_folder=tmp_folder)
    assert all([x.exists() for x in plot_files])
