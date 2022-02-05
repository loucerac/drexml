# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Unit testing for datasets module.
"""
import tempfile
from pathlib import Path
import shutil
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
from dreml.datasets import get_disease_data


def get_disease_path():
    """Get path to example disease env path.

    Returns
    -------
    pathlib.PosixPath
        Path to file.
    """
    with pkg_resources.path("dreml.resources", "experiment.env") as f:
        data_file_path = f
    return Path(data_file_path)

def prepare_disease():
    """Prepare fake disease data folder."""
    tmp_dir = Path(tempfile.mkdtemp())
    disease_path_in = get_disease_path()
    disease_path_out = tmp_dir.joinpath(disease_path_in)
    # Use as posix to make it compatible with python<=3.7
    shutil.copy(disease_path_in.as_posix(), disease_path_out.as_posix())


def test_get_disease_data():
    """Test get_disease_data."""

    disease_path = get_disease_path()
    gene_exp, pathvals, circuits, genes = get_disease_data(disease_path)

    assert gene_exp.to_numpy().ndim == 2
    assert pathvals.to_numpy().ndim == 2
    assert circuits.to_numpy().ndim == 2
    assert genes.to_numpy().ndim == 2
