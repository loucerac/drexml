# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Utilities module.
"""

from pathlib import Path

import pandas as pd
import pkg_resources
from sklearn.preprocessing import MinMaxScaler

from dreml.datasets import get_disease_data


def get_version():
    """Get DREML version."""
    return pkg_resources.get_distribution("dreml").version


def get_out_path(disease, debug):
    """Construct the path where the model must be saved.

    Returns
    -------
    pathlib.Path
        The desired path.
    """

    env_possible = Path(disease)

    if env_possible.exists() and (env_possible.suffix == ".env"):
        print(f"Working with experiment {env_possible.parent.name}")
        out_path = env_possible.parent.joinpath("ml")
    else:
        raise NotImplementedError("Use experiment")
    if debug:
        out_path.joinpath("debug")
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Storage folder: {out_path}")

    return out_path


def get_data(disease, debug, fmt="tsv.gz", scale=True):
    """Load disease data and metadata."""
    gene_xpr, pathvals, circuits, genes = get_disease_data(disease, fmt=fmt)

    if scale:

        pathvals = pd.DataFrame(
            MinMaxScaler().fit_transform(pathvals),
            columns=pathvals.columns,
            index=pathvals.index,
        )

    print(gene_xpr.shape, pathvals.shape)

    if debug:
        size = 9
        gene_xpr = gene_xpr.sample(n=size)
        pathvals = pathvals.sample(n=size)

    return gene_xpr, pathvals, circuits, genes
