#!/usr/bin/env python

import pathlib
from drexml.utils import convert_names
import sys
import pandas as pd


if __name__ == "__main__":
    _, folder = sys.argv
    folder = pathlib.Path(folder)

    for path in folder.rglob("shap_selection.tsv"):
        dataset = pd.read_csv(path, sep="\t", index_col=0)
        path_out = path.absolute().parent.joinpath(f"{path.stem}_symbol.tsv")
        dataset_out = convert_names(dataset, ["circuits", "genes"], axis=[0, 1])
        dataset_out.to_csv(path_out, sep="\t", index_label="circuit_name")

    for path in folder.rglob("shap_summary.tsv"):
        dataset = pd.read_csv(path, sep="\t", index_col=0)
        path_out = path.absolute().parent.joinpath(f"{path.stem}_symbol.tsv")
        dataset_out = convert_names(dataset, ["circuits", "genes"], axis=[0, 1])
        dataset_out.to_csv(path_out, sep="\t", index_label="circuit_name")

    for path in folder.rglob("stability_results.tsv"):
        dataset = pd.read_csv(path, sep="\t", index_col=0)
        path_out = path.absolute().parent.joinpath(f"{path.stem}_symbol.tsv")
        dataset_out = convert_names(dataset, ["circuits"], axis=[0])
        dataset_out.to_csv(path_out, sep="\t", index_label="circuit_name")
