# -*- coding: utf-8 -*-
"""
Postprocessing module.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from drexml.datasets import load_atc, load_drugbank


def fdr(pvalues):
    """Benjamini-Hochberg FDR p-value correction for multiple hypothesis testing.

    Parameters
    ----------
    pvalues : array-like
        Array of p-values.

    Returns
    -------
    array-like
        Array of corrected p-values.

    Notes
    -----
    This function is based on the `statsmodels.stats.multitest.multipletests` function.

    See Also
    --------
    statsmodels.stats.multitest.multipletests : Function for multiple hypothesis testing.

    Examples
    --------
    >>> import numpy as np
    >>> from drexml.postprocess import fdr
    >>> pvalues = np.array([0.05, 0.01, 0.001])
    >>> fdr(pvalues)
    array([0.05, 0.05, 0.05])

    """

    return multipletests(pvalues, alpha=0.05, method="fdr_bh")[1]


def test_ora_drugs_from_files(shap_sel_path, results_path, db_path=None, atc_path=None):
    """Helper function fo over representation analysis of ATC classification across
      levels for SHAP selected drugs.

    Parameters
    ----------
    shap_sel_path : str
        Path to the SHAP selected drug file.
    results_path : str
        Path to the results file.
    db_path : str, optional
        Path to the DrugBank file. If None, load from Zenodo.
    atc_path : str, optional
        Path to the ATC file. If None, load from Zenodo.

    Returns
    -------
    None.

    """

    shap_sel_path = Path(shap_sel_path)
    results_path = Path(results_path)

    sel_df = pd.read_csv(
        shap_sel_path,
        sep="\t",
        index_col=0,
    )

    if db_path is None:
        drugbank = load_drugbank()

    if atc_path is None:
        atc = load_atc()

    results = test_ora_drugs_from_frames(sel_df, drugbank, atc)

    results.to_csv(results_path, sep="\t", index=False)


def test_ora_drugs_from_frames(shap_selection_df, drugbank_df, atc_df):
    """Over representation analysis of ATC classification across levels for SHAP selected drugs.

    Parameters
    ----------
    shap_selection_df : pd.DataFrame
        SHAP selected drug frame.
    drugbank_df : pd.DataFrame
        DrugBank frame.
    atc_df : pd.DataFrame
        ATC frame.

    Returns
    -------
    pd.DataFrame
        Table with ORA pvalues for each drug and level (FDR-adjusted).
    """
    drugbank_df = drugbank_df.assign(
        is_selected=lambda x: x.symbol_id.isin(
            shap_selection_df.columns[shap_selection_df.any()]
        )
    )

    atc_level_to_len = {1: 1, 2: 3, 3: 4, 4: 5}

    results = []
    ora_min_len = 3

    for atc_level, level_len in atc_level_to_len.items():
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

        ora_dict = {}

        for atc_code in atc_dict.keys():
            drug_list_in_atc = np.unique(atc_dict[atc_code])
            n_drug_list_in_atc = drug_list_in_atc.size
            if n_drug_list_in_atc < ora_min_len:
                print(f"Ignore ATC codes with less than {ora_min_len} drugs")
            else:
                selected_drugs_in_atc = np.intersect1d(drugs_selected, drug_list_in_atc)
                selected_drugs_notin_atc = np.setdiff1d(
                    drugs_selected, drug_list_in_atc
                )

                drugs_in_atc_not_selected = np.intersect1d(
                    drug_list_in_atc, np.setdiff1d(background, drugs_selected)
                )
                drugs_notin_atc_not_selected = np.setdiff1d(
                    np.setdiff1d(background, drugs_selected), drug_list_in_atc
                )

                contingency_table = np.array(
                    [
                        [selected_drugs_in_atc.size, drugs_in_atc_not_selected.size],
                        [
                            selected_drugs_notin_atc.size,
                            drugs_notin_atc_not_selected.size,
                        ],
                    ]
                )

                odds_ratio, pvalue = stats.fisher_exact(
                    contingency_table, alternative="greater"
                )
                ora_dict[atc_code] = {
                    "ora_pval": pvalue,
                    "ora_unconditional_odds_ratio": odds_ratio,
                }

        this_results = (
            pd.DataFrame(ora_dict)
            .T.reset_index(names=["atc_code"])
            .assign(ora_bylevel_pval_adj=lambda x: fdr(x.ora_pval))
            .assign(atc_level=atc_level)
            .merge(atc_df, how="left")
        )

        results.append(this_results)

    col_order = [
        "atc_code",
        "atc_level",
        "atc_name",
        "ora_unconditional_odds_ratio",
        "ora_pval",
        "ora_bylevel_pval_adj",
        "ora_pval_adj",
    ]

    results = (
        pd.concat(results, axis=0, ignore_index=True)
        .assign(ora_pval_adj=lambda x: fdr(x.ora_pval))
        .loc[:, col_order]
    )

    return results
