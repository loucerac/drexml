# -*- coding: utf-8 -*-
"""
Test utilities module.
"""

import shutil
import tempfile
from pathlib import Path

from drexml.utils import get_resource_path

DATA_DICT = {
    "circuits": "circuits.tsv.gz",
    "gene_exp": "gene_exp.tsv.gz",
    "pathvals": "pathvals.tsv.gz",
    "genes": "genes.tsv.gz",
}


def make_disease_config(use_seeds=True, update=False):
    """Prepare fake disease data folder."""
    tmp_dir = Path(tempfile.mkdtemp())

    disease_path_out = tmp_dir.joinpath("experiment.env")
    with open(disease_path_out, mode="w", encoding="utf8") as this_file:
        pass

    with open(disease_path_out, "w", encoding="utf8") as this_file:
        for key, file_name in DATA_DICT.items():
            if use_seeds:
                if key == "circuits":
                    continue

            if not update:
                this_file.write(f"{key}=./{file_name}\n")

        if use_seeds:
            this_file.write("seed_genes=2180\n")

    for _, file_name in DATA_DICT.items():
        shutil.copy(
            get_resource_path(file_name).as_posix(),
            tmp_dir.joinpath(file_name).as_posix(),
        )

    return disease_path_out
