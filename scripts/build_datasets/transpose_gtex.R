#!/usr/bin/env Rscript
# Adaptaed from Adaptaed from https://github.com/babelomics/drexml-retinitis.git
args = commandArgs(trailingOnly=TRUE)

input_path <- file.path(args[1])
output_path <- file.path(args[2])

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

save_feather(
  t(trans_data),
  output_path
)
