#!/usr/bin/env Rscript
# Adaptaed from Adaptaed from https://github.com/babelomics/drexml-retinitis.git
args = commandArgs(trailingOnly=TRUE)

input_path <- file.path(args[1])
output_path <- file.path(args[2])
output_norm_path <- file.path(args[3])

#########################################
### Processing GTEx V8 datasets #####
#########################################

library(hipathia)
library(feather)
library(edgeR)
library(data.table)
library("R.utils")

save_feather <- function(x, path) {
  df <- data.frame(index = row.names(x), x)

  write_feather(df, path)
}

AnnotationHub::setAnnotationHubOption("ASK", FALSE)

## Read normalized gene expression
trans_data <- as.data.frame(feather::read_feather(input_path))
rownames(trans_data) <- trans_data[["index"]]
trans_data[["index"]] <- NULL

# scale data
exp_data <- normalize_data(as.matrix(trans_data))

###### Hipathia Processing #####
pathways <- load_pathways("hsa")

## Using Hipathia to compute the signal
results <- hipathia(exp_data, pathways, decompose = FALSE, verbose = FALSE)
path_vals <- get_paths_data(results, matrix = TRUE)
path_vals_norm <- normalize_paths(path_vals, pathways)

save_feather(
  t(path_vals),
  output_path
)

save_feather(
  t(path_vals_norm),
  output_norm_path
)
