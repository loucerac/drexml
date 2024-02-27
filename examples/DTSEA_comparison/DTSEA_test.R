###################################################################
####### COMPARISON DREXML DTSEA PACKAGE: 
####### https://cran.r-project.org/web/packages/DTSEA/vignettes/DTSEA.html
####### 14.02.2024 by Marina Esteban-Medina
###############################################################


if (!require(pacman, quietly = TRUE)){
  install.packages(pacman)
  library(pacman)
}else{
  library(pacman)
}


library(pacman)
p_load(magrittr, dplyr, BiocManager, devtools, here, ggplot2, tibble)


# Install the required packages for DTSEA and DisGeNet
BiocManager::install("fgsea")
devtools::install_github(c("hanjunwei-lab/DTSEA", "hanjunwei-lab/DTSEAdata"))
library(DTSEA)
install_bitbucket("ibi_group/disgenet2r", force = T)
library(disgenet2r)

##############################
########## 1. Loading seed signatures used in DREXML from DisGeNet DB
#############################

# disgenet_api_key <- "53a6cc82dbc7d2e0f3321afe8ca3160d63ad8fc9"
# Sys.setenv(DISGENET_API_KEY= disgenet_api_key)

# Load FA and FM gene signatures from DREXML (use cases for comparison with DTSEA)
#FA_signature <- disgenet2r::disease2gene(disease = "C0036341",database = "CURATED", score = c( 0.3,1 ), verbose = TRUE) ## Not working
FA_signature <- read.delim(here("examples", "DTSEA_comparison", "C0015625_disease_gda_summary.tsv"))  %>% filter(Score_gda >= 0.3) ## Downloaded from  disgenet website
FM_signature <- read.delim(here("examples", "DTSEA_comparison", "FM_ORPHANETgenes_in phy_paths.tsv"))

###############################
##### 2. Load and run DTSEA with their datasets on FA signatures: drug_target info and PPI graph
##############################
# Load the data from DTSEA https://github.com/hanjunwei-lab/DTSEAdata/tree/master
load(url("https://raw.githubusercontent.com/hanjunwei-lab/DTSEAdata/master/data/graph.rda"))
load(url("https://raw.githubusercontent.com/hanjunwei-lab/DTSEAdata/master/data/drug_targets.rda"))
colnames(drug_targets)<- gsub("drugbank_id", "drug_id", colnames(drug_targets)) ## Error from the DTSEA dataset



############
## DTSEA FA use-case
##############
# Perform a simple DTSEA analysis using default optional parameters then sort
# the result dataframe by normalized enrichment scores (NES)
set.seed(234)
result_FA <- DTSEA(network = graph,
                disease = FA_signature$Gene,
                drugs = drug_targets, verbose = FALSE) %>% arrange(desc(NES))

relevantFA_results <- select(result_FA, -leadingEdge) %>% arrange(desc(NES)) %>%  filter(NES > 0 & padj < .01)
NES_FAdrug_targets <- relevantFA_results[,c(1,3,6)] %>%  add_column(target = drug_targets$gene_target[match(relevantFA_results$drug_id, drug_targets$drug_id)]) %>%  
  add_column(drug = drug_targets$drug_name[match(relevantFA_results$drug_id, drug_targets$drug_id)]) 
length(unique(NES_FAdrug_targets$target)) ## 52 unique targets


############
## DTSEA: FM use-case
##############
# Perform a simple DTSEA analysis using default optional parameters then sort
# the result dataframe by normalized enrichment scores (NES)
set.seed(234)
result_FM <- DTSEA(network = graph,
                   disease = FM_signature$symbol,
                   drugs = drug_targets, verbose = FALSE) %>% arrange(desc(NES))

relevantFM_results <- select(result_FM, -leadingEdge) %>% arrange(desc(NES)) %>%  filter(NES > 0 & padj < .01)
NES_FMdrug_targets <- relevantFM_results[,c(1,3,6)] %>%  add_column(target = drug_targets$gene_target[match(relevantFM_results$drug_id, drug_targets$drug_id)]) %>%  
  add_column(drug = drug_targets$drug_name[match(relevantFM_results$drug_id, drug_targets$drug_id)])
length(unique(NES_FMdrug_targets$target)) ## 91 unique targets
 

###############################
##### 3. Load DREXML results from both uses-cases Fanconi Anemia (FA) and Familial Melanoma (FM) and compare with DTSEA results
##############################

###########
## Drexml-DTSEA FA comparison
###########

### 1. Load relevant filtered drexml results 
FA_drexml<- read.delim(gzfile(here("examples", "fanconi_anemia" , "results","shap_filtered_stability_symbol.tsv.gz"))) %>% column_to_rownames("circuit_name")

overlapFA <- colnames(FA_drexml)[colnames(FA_drexml) %in% NES_FAdrug_targets$target] ## 6 overlap

### 2. Rank targets in NES_FAdrug_targets based on p-value
nes_fa_ranks <- NES_FAdrug_targets  %>%
  mutate(rank_nes_fa = row_number()) %>%
  select(drug_id, drug ,target, padj, NES, rank_nes_fa) %>%
  filter(target %in% overlapFA)

### 3. Calculate mean absolute SHAP values and count relevant circuits for all targets
FA_stats <- sapply(FA_drexml, function(x) { # Exclude 'circuit_name' column
  c(mean_abs_shap = mean(abs(x), na.rm = TRUE), 
    circuits_above_zero = sum(abs(x) > 0, na.rm = TRUE))
})

  # Convert to a dataframe
FA_stats_df <- as.data.frame(t(FA_stats), stringsAsFactors = FALSE) %>%
  tibble::rownames_to_column("target") %>%
  mutate_at(vars(mean_abs_shap, circuits_above_zero), as.numeric) %>%
  arrange(desc(mean_abs_shap))

### 4. Create summarisation table

summary_DTSEAcomparisonFA <- merge(nes_fa_ranks, FA_stats_df, by = "target", all.x = TRUE)  %>% .[order(.$padj,decreasing = F),]
write.table( summary_DTSEAcomparisonFA, here("examples", "DTSEA_comparison" ,"summary_DTSEAcomparisonFA.tsv"), sep="\t", quote = F, row.names = F, col.names = T)

### 5. Compare with cmap results
FA_cmap <- read.delim(gzfile(here("examples", "cmap_pertubagens_comparison" ,"top_up_down_FA_cscoreL1000.tsv.gz"))) %>% select("pert_id", "pert_iname", "moa", "norm_cs", "single_target")

summary_CMAPcomparisonFA <- merge(FA_cmap, FA_stats_df, by.y = "target", by.x= "single_target", all.x = TRUE)  %>% .[order(.$norm_cs,decreasing = F),]
write.table( summary_CMAPcomparisonFA, here("examples", "cmap_pertubagens_comparison" ,"summary_CMAPcomparisonFA.tsv"), sep="\t",quote = F, row.names = F, col.names = T)


#######
##  Drexml-DTSEA FM comparison
#######

FM_drexml<- read.delim(gzfile(here("examples", "familial_melanoma" , "results" ,"shap_filtered_stability_symbol.tsv.gz"))) %>% column_to_rownames("circuit_name")

overlapFM <- colnames(FM_drexml)[colnames(FM_drexml) %in% NES_FMdrug_targets$target] 

### 2. Rank targets in NES_FAdrug_targets based on p-value
nes_fm_ranks <- NES_FMdrug_targets  %>%
  mutate(rank_nes_fm = row_number()) %>%
  select(drug_id, drug ,target, padj, NES, rank_nes_fm) %>%
  filter(target %in% overlapFM)

### 3. Calculate mean absolute SHAP values and count relevant circuits for all targets
FM_stats <- sapply(FM_drexml, function(x) { # Exclude 'circuit_name' column
  c(mean_abs_shap = mean(abs(x), na.rm = TRUE), 
    circuits_above_zero = sum(abs(x) > 0, na.rm = TRUE))
})

# Convert to a dataframe
FM_stats_df <- as.data.frame(t(FM_stats), stringsAsFactors = FALSE) %>%
  tibble::rownames_to_column("target") %>%
  mutate_at(vars(mean_abs_shap, circuits_above_zero), as.numeric) %>%
  arrange(desc(mean_abs_shap))

### 4. Create summarisation table

summary_DTSEAcomparisonFM <- merge(nes_fm_ranks, FM_stats_df, by = "target", all.x = TRUE) %>% .[order(.$padj,decreasing = F),]
write.table( summary_DTSEAcomparisonFM, here("examples", "DTSEA_comparison" ,"summary_DTSEAcomparisonFM.tsv"), sep="\t" ,quote = F, row.names = F, col.names = T)

### 5. Compare with cmap results
FM_cmap <- read.delim(gzfile(here("examples", "cmap_pertubagens_comparison" ,"top_up_down_FM_cscoreL1000.tsv.gz"))) %>% select("pert_id", "pert_iname", "moa", "norm_cs", "single_target")

summary_CMAPcomparisonFM <- merge(FM_cmap, FM_stats_df, by.y = "target", by.x= "single_target", all.x = TRUE)  %>% .[order(.$norm_cs,decreasing = F),]
write.table( summary_CMAPcomparisonFM, here("examples", "cmap_pertubagens_comparison" ,"summary_CMAPcomparisonFM.tsv"), sep="\t" ,quote = F, row.names = F, col.names = T)


