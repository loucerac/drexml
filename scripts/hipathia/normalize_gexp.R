#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

gtex_fname <- file.path(args[1])
output <- file.path(args[2])

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

  feather::write_feather(df, path)
}

AnnotationHub::setAnnotationHubOption("ASK", FALSE)

# Load  RNA-seq expression set from GTEx *.gct file.

## Read the downloaded GTEx raw counts dataset
expreset_raw <- fread(
  file = gtex_fname,
  header = T, sep = "\t"
) %>% as.data.frame(.)

rownames(expreset_raw) <- expreset_raw$Name
expreset_raw[c("Name", "Description")] <- list(NULL)
#expreset_raw <- expreset_raw[, -(1:2)]

# Normalization by TMM with "edgeR" package
dge <- DGEList(counts = expreset_raw)
tmm <- calcNormFactors(dge, method = "TMM")
logcpm <- cpm(tmm, prior.count = 3, log = TRUE)
trans_data <- translate_data(logcpm, "hsa")

save_feather(
  trans_data,
  output
)
