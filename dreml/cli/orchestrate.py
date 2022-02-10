#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for orchestrate.
"""

import click
import joblib

from dreml.utils import get_data, get_out_path, get_version


@click.command()
@click.option(
    "--debug/--no-debug", is_flag=True, default=False, help="Flag to run in debug mode."
)
@click.argument("disease-path", type=click.Path(exists=True))
@click.version_option(get_version())
def orchestrate(disease_path, debug):
    """[summary]"""

    print(f"running DREML orchestrate v {get_version()}")
    output_folder = get_out_path(disease_path)
    data_folder = output_folder.joinpath("tmp")
    data_folder.mkdir(parents=True, exist_ok=True)
    print(f"Tmp storage: {data_folder}")

    # Load data
    gene_xpr, pathvals, _, _ = get_data(disease_path, debug)
    joblib.dump(gene_xpr, data_folder.joinpath("features.jbl"))
    joblib.dump(pathvals, data_folder.joinpath("target.jbl"))
