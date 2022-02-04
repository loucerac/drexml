#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for orchestrate.
"""

import click

from dreml.utils import get_version

@click.command()
@click.option(
    "--download/--no-download", is_flag=True, default=False, help="Download data from zenodo."
)
@click.argument("disease_path", type=click.Path(exists=True))
@click.version_option(get_version())
def orchestrate(disease_path, download):
    """[summary]
    """

    print(f"running DREML orchestrate v {get_version()}")
    print(disease_path)
    print(download)

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    orchestrate()
