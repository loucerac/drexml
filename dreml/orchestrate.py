#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for orchestrate.
"""

import click
import pkg_resources

@click.command()
def orchestrate():
    """[summary]
    """
    version = pkg_resources.get_distribution('dreml').version
    print(f"running DREML orchestrate v {version}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    orchestrate()
