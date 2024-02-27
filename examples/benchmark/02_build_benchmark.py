#!/usr/bin/env python
# coding: utf-8

import pathlib
import dotenv
from drexml.utils import get_resource_path
import shutil

import pandas as pd

benchmark_folder = pathlib.Path(dotenv.find_dotenv()).parent
results_folder = benchmark_folder.joinpath("experiments")
results_folder.mkdir(exist_ok=True)
disease_template_env_path = benchmark_folder.joinpath("disease_template.env")

fpath = get_resource_path("circuit_names.tsv.gz")
circuit_names = pd.read_csv(fpath, sep="\t")
circuit_names.head()

for map_size in [1, 25, 50, 75, 100]:
    disease_name = f"disease_{map_size:03d}"
    disease_folder = results_folder.joinpath(disease_name)
    disease_folder.mkdir(exist_ok=True, parents=True)
    disease_map = circuit_names["circuit_id"][:map_size].to_frame()
    disease_map["in_disease"] = True
    disease_map.to_csv(disease_folder.joinpath("circuits.tsv.gz"), sep="\t", index=False)
    shutil.copyfile(disease_template_env_path, disease_folder.joinpath("disease.env"))
