# -*- coding: utf-8 -*-
"""
SHAP scores analysis for DREXML.
"""
import pandas as pd
from sklearn.preprocessing import maxabs_scale


def process_data(data_folder, stability_threshold=0.6):
    """
    Once the folder where the model results is defined, this function reads the shap_summary (scores) and shap_selection (selected as relevant) .tsv     files into DataFrames, applies Min-Max normalization to make the scores comparable, filters the circuits
    based on a desired stability threshold (by default 0.6), and saves the final shap_filtered_stable dataframe to a .tsv file.

    Args:
    data_folder (str): The path to the data folder.
    stability_threshold (float): The stability threshold to be set (we recommend using a value between 0.4-0.7)

    Returns:
    None. The final dataframe is saved as a .tsv file.
    """

    # Read the .tsv files into a DataFrames (circuits X genes)
    shap_values = pd.read_csv(
        f"{data_folder}/shap_summary_symbol.tsv", sep="\t", index_col=0
    )
    shap_selection = pd.read_csv(
        f"{data_folder}/shap_selection_symbol.tsv", sep="\t", index_col=0
    )

    # Read the .tsv files into a DataFrames (circuits X metrics)
    stability = pd.read_csv(
        f"{data_folder}/stability_results_symbol.tsv", sep="\t", index_col=0
    )

    shap_selection = shap_selection.loc[:, shap_selection.any()]

    circuits_stable = stability[
        stability["stability"] > stability_threshold
    ].index.intersection(shap_values.index)

    # circuit-wise scale by max abs value
    # filter using the selected genes (columns)
    # filter using stable circuits
    filtered_df_stable = (
        shap_values.transform(maxabs_scale)
        .multiply(shap_selection)
        .dropna(axis=1)
        .loc[circuits_stable]
    )

    filtered_df_stable.to_csv(
        f"{data_folder}/shap_filtered_stability_symbol.tsv",
        sep="\t",
        index=True,
        index_label="circuit_name",
    )
