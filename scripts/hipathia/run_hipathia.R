#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

gtex_fname <- args[1]
vers <- args[2]
output <- args[3]

#########################################
### Processing GTEx V8 datasets #####
#########################################

library(hipathia)
library(feather)
library(edgeR)
library(data.table)
library("R.utils")

AnnotationHub::setAnnotationHubOption("ASK", FALSE)

dir.create(here("data", "interim"), showWarnings = FALSE, recursive=TRUE)
dir.create(here("data", "final"), showWarnings = FALSE, recursive=TRUE)


# Load  RNA-seq expression set from GTEx *.gct file.

## Read the downloaded GTEx raw counts dataset
expreset_raw <- fread(
  file = here("data", "raw", gtex_fname),
  header = T, sep = "\t"
) %>% as.data.frame(.)

rownames(expreset_raw) <- expreset_raw$Name
expreset_raw <- expreset_raw[, -(1:2)]
print("read...done")

# Normalization by TMM with "edgeR" package
dge <- DGEList(counts = expreset_raw)
print("dge...done")
tmm <- calcNormFactors(dge, method = "TMM")
print("tmm...done")
logcpm <- cpm(tmm, prior.count = 3, log = TRUE)
print("dge...done")

# eliminate from rownames the ".number", beacuse Hipathia does not process them well
# rownames(logcpm) <- gsub("\\..*", "", rownames(logcpm))
print("normalization 1...done")


###### Hipathia Processing #####

trans_data <- translate_data(logcpm, "hsa")
exp_data <- normalize_data(trans_data)

## Loading Pathways (only physiological)

# physiological_pathways
#path_list <- read.table(file = here("data", "raw", "physiological_paths.tsv"), sep = "\t")
pathways <- load_pathways("hsa")

## Using Hipathia to compute the signal

results <- hipathia(exp_data, pathways, decompose = FALSE, verbose = FALSE)
path_vals <- get_paths_data(results, matrix = TRUE)
path_vals_norm <- normalize_paths(path_vals, pathways)

save_feather <- function(x, path) {
  df <- data.frame(index = row.names(x), x)

  feather::write_feather(df, path)
}

edger_vers <- paste0("v", packageVersion("edgeR"))
hipathia_vers <- paste0("v", packageVersion("hipathia"))

save_feather(
  t(exp_data),
  here("data", "final", paste0("gexp_gtex-", vers, "_edger-", edger_vers, ".feather"))
)

save_feather(
  t(path_vals),
  here("data", "final", paste0("pathvals_gtex-", vers, "_edger-", edger_vers, "_hipathia-", hipathia_vers, ".feather"))
)

save_feather(
  t(path_vals_norm),
  here("data", "final", paste0("pathvals_gtex-", vers, "_edger-", edger_vers, "_hipathia-norm-", hipathia_vers, ".feather"))
)

