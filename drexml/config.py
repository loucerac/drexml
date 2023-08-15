# -*- coding: utf-8 -*-
"""
Config module.
"""


DEFAULT_DICT = {
    "seed_genes": None,
    "disease_id": None,
    "use_physio": "true",
    "gene_exp": None,
    "gene_exp_zenodo": False,
    "pathvals": None,
    "pathvals_zenodo": False,
    "circuits": None,
    "circuits_zenodo": False,
    "genes": None,
    "genes_zenodo": False,
    "activity_normalizer": "false",
    "circuits_column": "in_disease",
    "genes_column": "drugbank_approved_targets",
    "GTEX_VERSION": "v8",
    "MYGENE_VERSION": "v20230220",
    "DRUGBANK_VERSION": "v050110",
    "HIPATHIA_VERSION": "v2-14-0",
    "EDGER_VERSION": "v3-40-0",
}


VERSION_DICT = {
    "GTEX_VERSION": ["v8"],
    "MYGENE_VERSION": ["v20230220"],
    "DRUGBANK_VERSION": ["v050110"],
    "HIPATHIA_VERSION": ["v2-14-0"],
    "EDGER_VERSION": ["v3-40-0"],
}
