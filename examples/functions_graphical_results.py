## Functions for Visualization: Graphical Insights into DRExM³L Package Results ##

# -*- coding: utf-8 -*-
"""
SHAP scores filtering for DREXML results.
"""



def process_data(data_folder, stability_threshold=0.6):
    """
    Once the folder where the model results is defined, and the files are decompressed
    this function reads the shap_summary (KDTs scores) and shap_selection (KDTs selected as relevant) .tsv
    files into DataFrames, applies Min-Max normalization to make the scores comparable, filters the circuits
    based on a desired stability threshold (by default 0.6), and saves the final shap_filtered_stable data frame to a .tsv file.

    Args:
    data_folder (str): The path to the data folder.
    stability_threshold (float): The stability threshold to be set (we recommend using a value between 0.4-0.7)

    Returns:
    None. The final data frame is saved as a .tsv file.
    """
    import pandas as pd
    from sklearn.preprocessing import maxabs_scale
    
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
    # filtered_df_stable = (
    #     shap_values.transform(maxabs_scale, axis=1)
    #     .multiply(shap_selection)
    #     .dropna(axis=1)
    #     .loc[circuits_stable]
    # )

    # Scale each row by its maximum absolute value
    scaled_shap_values = shap_values.apply(lambda x: x / x.abs().max(), axis=1)
    
    # # Filter based on the stability threshold and shap_selection
    filtered_df_stable = (
        scaled_shap_values.multiply(shap_selection)
        .dropna(axis=1)
        .loc[circuits_stable]        
    )

    # Remove columns with all 0 values
    filtered_df_stable = filtered_df_stable.loc[:, (filtered_df_stable != 0).any(axis=0)]

    filtered_df_stable.to_csv(
        f"{data_folder}/shap_filtered_stability_symbol.tsv",
        sep="\t",
        index=True,
        index_label="circuit_name",
    )
    
    

def create_heatmap_KDTsCir_annot(data_folder, assets_folder, filtered_file_name):

    """
    This function reads the filtered shap_relevant_table data generated from the process_data() function  and the annotations 
    of the most common drug-effect over the drug-targets. Then, it creates a heatmap with the drug-targets (annotated with the most
    common drug effect on top) as columns and the circuits as rows, filled  with the SHAP scores obtained. 
    The heatmap is saved to a .pdf file on the same data_folder.

    Args:
    data_folder (str): The path to the data folder.
    assets_folder (str): The path to the assets folder.
    filtered_file_name (str): The name of the shap scores filtered file.

    Returns:
    None. The heatmap is saved as a .pdf file.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Load the previous filtered table
    filtered_df_stable = pd.read_csv(f"{data_folder}/{filtered_file_name}", sep='\t', index_col=0)

    # Load annotations of the most common drug-effects over the drug-targets
    annots = pd.read_csv(f"{assets_folder}/drugbank-v050110_mostdrugeffects_KDTs.tsv", sep='\t', index_col=0)

    # Subset the relevant drug-targets
    annots = annots.loc[annots.index.intersection(filtered_df_stable.columns),]

    drug_action_colors = {
        "Inhibitor": "magenta",
        "Activator": "cyan",
        "Ligand": "yellow",
        "other": "grey",
        "Modulator": "orange"
    }
    drug_action_colors_list = [drug_action_colors[action] for action in annots["drug_action"]]

    # If the number of circuits is greater than 150, group by pathways
    if filtered_df_stable.shape[0] > 150:
        # Split the circuit names by ": " to get the pathways
        filtered_df_stable.index = filtered_df_stable.index.str.split(": ").str[0]

        # Aggregate by pathway
        filtered_df_stable = filtered_df_stable.groupby(level=0).mean()

        # Calculate the figure size and the font size based on the number of rows and columns
        num_rows, num_cols = filtered_df_stable.shape
        scale_factor = max(num_rows, num_cols) ** 0.5  # adjust this as needed
        
        fig_width = max(50, num_cols / scale_factor)  # adjust this as needed
        fig_height = max(30, num_rows / scale_factor)  # adjust this as needed
        # font_size = min(5, 300 / max(num_rows, num_cols))  # adjust this as needed

       
        # Create the clustermap with adjusted size
        cluster_map = sns.clustermap(
            filtered_df_stable, cmap="RdBu_r", 
            col_colors=[drug_action_colors_list], 
            linewidths=0.5, cbar_kws={'label': 'SHAP Value'}, 
            figsize=(fig_width,fig_height),
            dendrogram_ratio=(.1, .2),
            
        )
        # cluster_map.ax_heatmap.tick_params(axis='both', which='major', labelsize= 0.8)
        
        # Add titles to the x and y axes
        cluster_map.ax_heatmap.set_xlabel('Drug-target', fontsize=25)
        cluster_map.ax_heatmap.set_ylabel('Pathway name', fontsize=25)


        print("The number of circuits is greater than 150. The heatmap will be created using pathways.")
    else:
        print("The number of circuits is less than or equal to 150. The heatmap will be created using circuits.")

        # Calculate the figure size and the font size based on the number of rows and columns
        num_rows, num_cols = filtered_df_stable.shape
        scale_factor = max(num_rows, num_cols) ** 0.5  # adjust this as needed
        
        fig_width = max(50, num_cols / scale_factor)*1  # adjust this as needed
        fig_height = max(60, num_rows / scale_factor)*1  # adjust this as needed
        # font_size = min(18, 500 / max(num_rows, num_cols))  # adjust this as needed
        
        sns.set(font_scale= 1.5) # Adjust the font size if necessary

        # Create the clustermap with fixed size
        cluster_map = sns.clustermap(
            filtered_df_stable,
            cmap="RdBu_r", 
            col_colors=[drug_action_colors_list],
            linewidths=0.5, 
            cbar_kws={'label': 'SHAP Value'},
            figsize=(fig_width,fig_height),
            dendrogram_ratio=(.13, .2)
        )
        
        # Add titles to the x and y axes
        cluster_map.ax_heatmap.set_xlabel('Drug-target', fontsize=75)
        cluster_map.ax_heatmap.set_ylabel('Pathway name', fontsize=75)

        # Adjust the label size
        plt.setp(cluster_map.ax_heatmap.get_xticklabels(), size=30)  # Adjust x labels size
        plt.setp(cluster_map.ax_heatmap.get_yticklabels(), size=30)  # Adjust y labels size
        

    # Set larger font size for the colorbar label and ticks
    cbar = cluster_map.ax_cbar
    cbar.set_ylabel('SHAP Value', fontsize= 50)  # Set a larger font size for the label
    cbar.tick_params(labelsize= 45)  # Set a larger font size for the ticks
       

    # Add a title to the color legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=action, markersize=20)
        for action, color in drug_action_colors.items()
    ]
    legend = plt.legend(
        handles=legend_elements,
        title="Drug Action", 
        loc='upper left', 
        bbox_to_anchor=(18,1),
        fontsize=65,
        title_fontsize=70,
        markerscale=3
    )
    legend.get_frame().set_facecolor('white')

    # Add a title to the entire figure
    plt.suptitle('Heatmap of Disease Map and relevant Drug-targets', fontsize=80, y=1.05)
  
    # Save the plot
    plt.savefig(f"{data_folder}/heatmap_circuits_KDT_drugeff.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    
def create_histograms(data_folder, filtered_file_name):
    """
    This function reads the filtered data, creates histograms from the SHAP scores of each drug-target, 
    and saves the histograms to a composed .pdf file.

    Args:
    data_folder (str): The path to the data folder.
    filtered_file_name (str): The name of the  shap scores filtered file.

    Returns:
    None. The histograms are saved as a .pdf file.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

   # Load the previous filtered table
    data = pd.read_csv(f"{data_folder}/{filtered_file_name}", sep='\t', index_col=0)

    # Number of columns in the data
    num_cols = len(data.columns[1:])  

    # Calculate the number of rows required
    num_rows = int(np.ceil(num_cols / 4.0))  

    # Set up the figure and axes
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 3*num_rows))

    # Flatten the axes
    axes = axes.flatten()

    # Remove the extra subplots
    for i in range(num_cols, num_rows*4):  
        fig.delaxes(axes[i])

    # Create histograms for all score columns
    for i, col in enumerate(data.columns[1:]):
        sns.histplot(data[col], ax=axes[i], kde=True)
        axes[i].set_title(col)
        axes[i].set_xlabel('')  # remove the x-label

    # Improve layout
    plt.tight_layout()
    plt.savefig(f"{data_folder}/relevantDrugTargets_scores.pdf", dpi=300)
    plt.show()
    
    
    
def create_boxplots(data_folder, filtered_file_name):
    """
    This function reads the filtered data, creates boxplots for all score columns, and saves the boxplots to a .pdf file.

    Args:
    data_folder (str): The path to the data folder.
    filtered_file_name (str): The name of the shap scores filtered file.

    Returns:
    The boxplots are saved as a .pdf file.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load the previous filtered table
    data = pd.read_csv(f"{data_folder}/{filtered_file_name}", sep='\t', index_col=0)

    # Melt the DataFrame to make it suitable for boxplots
    data.reset_index(inplace=True) ## Reset the index back to a column
    melted_data = data.melt(id_vars = "circuit_name")

    # Compute median of each column
    medians = melted_data.groupby("variable")["value"].median().sort_values(ascending=False)

    # Order the data based on medians
    melted_data["variable"] = pd.Categorical(melted_data["variable"], categories=medians.index, ordered=True)
    
    # Calculate the width size based on the number of variables
    num_variables = melted_data["variable"].nunique()
    fig_width = max(10, num_variables / 2)  # adjust this formula as needed
    

    # Create the boxplots
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.figure(figsize=(fig_width, 12), facecolor='white')
    sns.set_style("whitegrid")  # set Seaborn style to white grid
    boxplot = sns.boxplot(x="variable", y="value", data = melted_data)
    
    # Set the font size for x and y  tick labels
    boxplot.set_xticklabels(boxplot.get_xticklabels(), size=22)
    boxplot.tick_params(axis='y', labelsize=22)
    
    plt.xticks(rotation=40, ha = "right")
    plt.title("Boxplots of SHAP-Scores", fontsize=50, y=1.05)
    plt.ylabel('SHAP Scores', fontsize=35)
    plt.xlabel('Drug-target', fontsize=35)
    plt.tight_layout()
    plt.savefig(f"{data_folder}/relevant_drugTargetscores_boxplots.pdf", dpi=300)
    plt.show()
    
    
def create_heatmap_drugsCircuit_top30KDTs(data_folder, filtered_file_name, assets_folder):
    """
    This function reads the filtered matrix of shap scores (columns=drug-targets, rows = disease-circuits) and the drug-target-effects data, 
    filters the top 10 best-scored drug-targets from SHAP and creates a heatmap with the drugs from the top 10 scored targets and the circuits.
    If the number of circuits is greater than 50, the circuits are grouped by their pathways (the first part of the circuit name before ": ").
    
    Args:
    data_folder (str): The path to the data folder where all results from the DREXML model are.
    filtered_file_name (str): The name of the filtered data file.
    assets_folder (str): The path to the assets folder where the drug-target interactions file is located.
    
    Returns:
    A heatmap is displayed and saved in the data folder.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Read the genes data
    df = pd.read_csv(f"{data_folder}/{filtered_file_name}", sep='\t', index_col=0)

    # Calculate the mean of absolute values for each gene
    mean_scores = df.abs().mean()

    # Get the top 30 genes by mean score
    top_genes = mean_scores.nlargest(30)

    # Read the drugs data
    df_drugs = pd.read_csv(f"{assets_folder}/drugbank-v050110_alldrugbyaction.tsv", sep='\t', skiprows=1, names=["Drug", "Action", "KDT", "Gene", "Drug_effect", "drugKDT"])

    # Filter the drugs that target the top 10 genes
    df_drugs_top_genes = df_drugs[df_drugs['Gene'].isin(top_genes.index)]

    # Create a new dataframe where each row corresponds to a drug and a circuit, and the value is the mean score of the top gene targeted by the drug in that circuit
    df_heatmap = pd.DataFrame(index=df.index.unique(), columns=df_drugs_top_genes['Drug'].unique())

    # Populate the dataframe with mean scores
    for circuit in df_heatmap.index:
        for drug in df_heatmap.columns:
            # Get the genes targeted by the drug
            genes = df_drugs_top_genes.loc[df_drugs_top_genes['Drug'] == drug, 'Gene'].values
            
            # Calculate the mean score of the genes in the circuit
            df_heatmap.loc[circuit, drug] = df.loc[circuit, genes].abs().mean()

    # Convert the scores to floats
    df_heatmap = df_heatmap.astype(float)

    # If the number of circuits is greater than 50, group by pathways
    if df_heatmap.shape[0] > 50:
        # Split the circuit names by ": " to get the pathways
        df_heatmap.index = df_heatmap.index.str.split(": ").str[0]

        # Aggregate by pathway
        df_heatmap = df_heatmap.groupby(level=0).mean()

        print("The number of circuits is greater than 50. The heatmap will be created using pathways.")
    else:
        print("The number of circuits is less than or equal to 50. The heatmap will be created using circuits.")
        # sns.set(font_scale= 2) # Adjust the font size if necessary
        # plt.ylabel('Circuit')

    # Calculate the figure size and the font size based on the number of rows and columns
    num_rows, num_cols = df_heatmap.shape
    scale_factor = max(num_rows, num_cols) ** 0.5  # adjust this as needed
    font_size = 10.0 / scale_factor  # adjust this as needed
    sns.set(font_scale= font_size) 
    
    fig_width = max(40, num_cols / scale_factor)  # adjust this as needed
    fig_height = max(30, num_rows / scale_factor)  # adjust this as needed
   
    # Plot a heatmap
    cluster_map = sns.clustermap(df_heatmap, cmap="Blues", 
                                 figsize=(fig_width, fig_height), # adjusted figsize here to control the size of the plot directly
                                 cbar_kws={'label': 'Drug score'},
                                 linewidth =0.01,
                                 dendrogram_ratio=(.12, .22),
                                )
    # Add titles to the x and y axes
    cluster_map.ax_heatmap.set_xlabel('Drug-name', fontsize=45)
    cluster_map.ax_heatmap.set_ylabel('Pathway name', fontsize=43)

    # Adjust the label size
    plt.setp(cluster_map.ax_heatmap.get_xticklabels(), size= 34)  # Adjust x labels size
    plt.setp(cluster_map.ax_heatmap.get_yticklabels(), size=45)  # Adjust y labels size

    # Set larger font size for the colorbar label and ticks
    cbar = cluster_map.ax_cbar
    cbar.set_ylabel('Drug score', fontsize= 40)  # Set a larger font size for the label
    cbar.tick_params(labelsize= 38)  # Set a larger font size for the ticks

    
    plt.suptitle('Mean scores of the drugs from the top 30 best-scored drug-targets over the Disease Map ', fontsize=50, y=1.05)
    # plt.tight_layout()
    plt.savefig(f"{data_folder}/heatmap_top30KDTdrugs_circuit.pdf", dpi=300, bbox_inches='tight')
    plt.show()
