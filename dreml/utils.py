#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Utilities module.
"""

import pkg_resources

def get_version():
    """Get DREML version.
    """
    return pkg_resources.get_distribution('dreml').version
