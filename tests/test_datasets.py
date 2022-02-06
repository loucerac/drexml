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


def prepare_disease():
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


def test_get_disease_data():
    """Test get_disease_data."""

    disease_path = get_resource_path("experiment.env")
    disease_path = prepare_disease()
    gene_exp, pathvals, circuits, genes = get_disease_data(disease_path)

    assert gene_exp.to_numpy().ndim == 2
    assert pathvals.to_numpy().ndim == 2
    assert circuits.to_numpy().ndim == 2
    assert genes.to_numpy().ndim == 2
