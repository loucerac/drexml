# -*- coding: utf-8 -*-
"""
SHAP scores analysis for DREXML.
"""
import pandas as pd


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
    # Read the .tsv files into a DataFrames
    shap_values = pd.read_csv(
        f"{data_folder}/shap_summary_symbol.tsv", sep="\t", index_col=0
    )
    shap_selection = pd.read_csv(
        f"{data_folder}/shap_selection_symbol.tsv", sep="\t", index_col=0
    )
    stability = pd.read_csv(
        f"{data_folder}/stability_results_symbol.tsv", sep="\t", index_col=0
    )

    # Define min and max range for normalization
    min_range = -1
    max_range = 1

    # Apply Min-Max normalization to each row
    shap_normalized = shap_values.copy()
    shap_normalized.iloc[:, 1:] = shap_normalized.iloc[:, 1:].apply(
        lambda row: (row - row.min())
        / (row.max() - row.min())
        * (max_range - min_range)
        + min_range,
        axis=1,
    )

    shap_filtered_norm = shap_normalized * shap_selection
    filtered_df = shap_filtered_norm.loc[:, (shap_filtered_norm != 0).sum() != 0]

    # Now filter those circuits that have stability above the desired threshold
    stability_00 = stability[stability["stability"] > stability_threshold]

    stability_00.reset_index(inplace=True)
    filtered_df.reset_index(inplace=True)

    filtered_df_stable = filtered_df[
        filtered_df["circuit_name"].isin(stability_00["circuit_name"])
    ]
    filtered_df_stable.to_csv(
        f"{data_folder}/shap_filtered_stability_symbol.tsv", sep="\t", index=False
    )
