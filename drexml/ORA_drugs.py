# -*- coding: utf-8 -*-
"""
Over representation analysis of ATC classifiction across levels for SHAP selected drugs.
"""

import pathlib
import shutil
import urllib.request

import dotenv
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def fdr(pvalues):
    """
    Function to apply Benjamini-Hochberg FDR p-value correction for multiple hypothesis testing.

    Args:
    pvalues (array-like): array of p-values to correct.

    Returns:
    array: corrected p-values.
    """
    return multipletests(pvalues, alpha=0.05, method="fdr_bh")[1]


def main(db_path, atc_path):
    """
    Main function that reads drugbank data and ATC codes, applies over-representation analysis
    on selected drugs at different ATC levels, and saves the results.

    Args:
    db_path (str): Path to the drugbank data file.
    atc_path (str): Path to the ATC codes file.

    Returns:
    None
    """
    db_path = pathlib.Path(db_path)
    atc_path = pathlib.Path(atc_path)
    np.random.seed(42)

    project_root = pathlib.Path(dotenv.find_dotenv()).absolute().parent
    data_folder = project_root.joinpath("data")
    data_folder.joinpath("raw")
    final_folder = data_folder.joinpath("final")
    results_folder = project_root.joinpath("results")
    tables_folder = results_folder.joinpath("tables")
    tables_folder.mkdir(parents=True, exist_ok=True)

    atc_url = "https://raw.githubusercontent.com/fabkury/atcd/master/WHO%20ATC-DDD%202021-12-03.csv"

    # If ATC codes file doesn't exist, download it from the internet
    if not atc_path.exists():
        with urllib.request.urlopen(atc_url) as response, open(
            atc_path, "wb"
        ) as out_file:
            shutil.copyfileobj(response, out_file)

    atc_code_name = pd.read_csv(atc_path, usecols=["atc_code", "atc_name"])

    # Read data files
    shap_selection_df = pd.read_csv(
        results_folder.joinpath("ml", "shap_selection_symbol.tsv"),
        sep="\t",
        index_col=0,
    )
    drugbank_df = pd.read_csv(final_folder.joinpath(db_path), sep="\t").assign(
        is_selected=lambda x: x.symbol_id.isin(
            shap_selection_df.columns[shap_selection_df.any()]
        )
    )

    atc_level_to_len = {1: 1, 2: 3, 3: 4, 4: 5}

    ora_min_len = 3

    # Apply over-representation analysis on selected drugs at different ATC levels
    for atc_level, level_len in atc_level_to_len.items():
        # Preprocess the data for ORA
        tmp_df = (
            drugbank_df.loc[:, ["drugbank_id", "atc_codes", "is_selected"]]
            .dropna()
            .drop_duplicates()
            .groupby("drugbank_id")
            .agg({"atc_codes": list, "is_selected": "any"})
            .reset_index()
            .explode("atc_codes")
            .assign(atc_codes=lambda x: x.atc_codes.str.split("|"))
            .explode("atc_codes")
            .assign(atc_codes=lambda x: x.atc_codes.str[:level_len])
            .drop_duplicates()
        )
        atc_dict = (
            tmp_df[["drugbank_id", "atc_codes"]]
            .groupby("atc_codes")
            .agg({"drugbank_id": list})
            .to_dict()["drugbank_id"]
        )
        background = tmp_df.drugbank_id.unique()
        drugs_selected = tmp_df.drugbank_id[tmp_df.is_selected].unique()

        # Apply ORA
        for atc_code in atc_dict.keys():
            drug_list_in_atc = np.unique(atc_dict[atc_code])
            n_drug_list_in_atc = drug_list_in_atc.size
            if n_drug_list_in_atc < ora_min_len:
                print(f"Ignore ATC codes with less than {ora_min_len} drugs")
            else:
                np.intersect1d(drugs_selected, drug_list_in_atc)
                np.setdiff1d(drugs_selected, drug_list_in_atc)

                drugs_in_atc_not_selected = np.intersect1d(
                    drug_list_in_atc, np.setdiff1d(background, drugs_selected)
                )
                drugs_notin_atc_not_selected = np.setdiff1d(
                    np.setdiff1d(background, drugs_selected), drug_list_in_atc
                )
