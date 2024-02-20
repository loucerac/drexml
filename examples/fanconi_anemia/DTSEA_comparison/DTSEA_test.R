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
p_load(magrittr, dplyr, BiocManager, devtools, here)


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
FA_signature <- read.delim(here("examples","fanconi_anemia", "DTSEA_comparison", "C0015625_disease_gda_summary.tsv"))  %>% filter(Score_gda >= 0.3) ##Downloaded from website


###############################
##### 2. Load and run DTSEA with their datasets on FA signatures: drug_target info and PPI graph
##############################
# Load the data from DTSEA https://github.com/hanjunwei-lab/DTSEAdata/tree/master
load(url("https://raw.githubusercontent.com/hanjunwei-lab/DTSEAdata/master/data/graph.rda"))
load(url("https://raw.githubusercontent.com/hanjunwei-lab/DTSEAdata/master/data/drug_targets.rda"))
colnames(drug_targets)<- gsub("drugbank_id", "drug_id", colnames(drug_targets)) ## Error from the DTSEA dataset

# data("example_disease_list", package = "DTSEA")
# data("example_drug_target_list", package = "DTSEA")
# data("example_ppi", package = "DTSEA")


# Perform a simple DTSEA analysis using default optional parameters then sort
# the result dataframe by normalized enrichment scores (NES)
result <- DTSEA(network = graph,
                disease = FA_signature$Gene,
                drugs = drug_targets, verbose = FALSE) %>% arrange(desc(NES))

relevant_results <- select(result, -leadingEdge) %>% arrange(desc(NES)) %>%  filter(NES > 0 & padj < .01)
NES_drug_targets <- relevant_results[,c(1,2,6)] %>%  add_column(target = drug_targets$gene_target[match(relevant_results$drug_id, drug_targets$drug_id)]) 
length(unique(NES_drug_targets$target)) ## 52 unique targets
 

###############################
##### 3. Load DREXML results and compare
##############################

FA_drexml<- read.delim(gzfile("/home/m3m/INFO_PROYECTO/drexml/examples/fanconi_anemia/results/shap_selection_symbol.tsv.gz")) %>% column_to_rownames("circuit_name")

FA_kdts_drexml <- FA_drexml[, apply(FA_drexml,2, function(x) any(x == "True"))] ## 93 unique targets

overlap <- colnames(FA_kdts_drexml)[colnames(FA_kdts_drexml) %in% NES_drug_targets$target] ## 7 overlap

NES_drug_targets_overlap <- filter(NES_drug_targets, target %in% overlap )



