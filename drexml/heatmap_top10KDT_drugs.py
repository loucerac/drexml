# -*- coding: utf-8 -*-
"""
Analysis of top 10 drug-targets by their drugs on a Heatmap for DREXML.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def create_heatmap_drugsCircuit_top10KDTs(KDTs_path, drugs_path, data_path):
    """
    This function reads the filtered matrix of shap scores (columns=drug-targets, rows = disease-circuits) and the drug-target-effects data, 
    filters the top 10 best-scored drug-targets from SHAP and creates a heatmap with the drugs from the top 10 scored targets and the circuits.
    
    Args:
    KDTs_path (str): The path to the shap_scores data file.
    drugs_path (str): The path to the drug-target interactions file.
    data_path (str): The path to the data folder where all results from the DREXML model are.
    
    
    Returns:
    A heatmap is displayed and saved in the data_path.
    """

    # Read the genes data
    df = pd.read_csv(KDTs_path, sep='\t', index_col="circuit_name")

    # Calculate the mean of absolute values for each gene
    mean_scores = df.abs().mean()

    # Get the top 10 genes by mean score
    top_10_genes = mean_scores.nlargest(10)

    # Read the drugs data
    df_drugs = pd.read_csv(drugs_path, sep='\t', skiprows=1, names=["Drug", "Action", "KDT", "Gene", "Drug_effect", "drugKDT"])

    # Filter the drugs that target the top 10 genes
    df_drugs_top_10_genes = df_drugs[df_drugs['Gene'].isin(top_10_genes.index)]

    # Create a new dataframe where each row corresponds to a drug and a circuit, and the value is the mean score of the top gene targeted by the drug in that circuit
    df_heatmap_10 = pd.DataFrame(index=df.index.unique(), columns=df_drugs_top_10_genes['Drug'].unique())

    # Populate the dataframe with mean scores
    for circuit in df_heatmap_10.index:
        for drug in df_heatmap_10.columns:
            # Get the genes targeted by the drug
            genes = df_drugs_top_10_genes.loc[df_drugs_top_10_genes['Drug'] == drug, 'Gene'].values
            
            # Calculate the mean score of the genes in the circuit
            df_heatmap_10.loc[circuit, drug] = df.loc[circuit, genes].abs().mean()

    # Convert the scores to floats
    df_heatmap_10 = df_heatmap_10.astype(float)

    # Plot a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_heatmap_10, cmap="Blues", center=0)
    plt.title('Mean scores of top 10 genes targeted by each drug in each circuit')
    plt.xlabel('Drug')
    plt.ylabel('Circuit')
    plt.savefig(f"{data_path}/heatmap_top10KDTdrugs_circuit.pdf", dpi=300)
    plt.show()