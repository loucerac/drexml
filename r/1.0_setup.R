rm(list = ls())

library(here)
library(dotenv)
dotenv::load_dot_env(file = here(".env"))
DATA_PATH <- Sys.getenv("DATA_PATH")

genes_fname <- "genes.rds"
pathvals_fname <- "expreset_pathvals.rds"
expression_fname <- "expreset_Hinorm.rds"
circuits_fname <- "circuits.rds"
fanconi_circuits_fname <- "circuits_FA.rds"
fanconi_pathvals_fname <- "expreset_pathvals_FA.rds"

genes <- readRDS(file.path(DATA_PATH, genes_fname))
rownames(genes) <- genes$Entrezs
genes$Entrezs <- NULL
expr <- readRDS(file.path(DATA_PATH, expression_fname))
pathvals <- readRDS(file.path(DATA_PATH, pathvals_fname))
circuits <- readRDS(file.path(DATA_PATH, circuits_fname))
rownames(circuits) <- circuits$hipathia

fanconi_pathvals <- readRDS(file.path(DATA_PATH, fanconi_pathvals_fname))
fanconi_circuits <- readRDS(file.path(DATA_PATH, fanconi_circuits_fname))
rownames(fanconi_circuits) <- fanconi_circuits$hipathia


save_feather <- function(x, path) {
    df <- data.frame(index = row.names(x), x)

    feather::write_feather(df, path)
}

save_feather(
    as.data.frame(t(pathvals)),
    file.path(DATA_PATH, paste0(pathvals_fname, ".feather"))
)

save_feather(
    as.data.frame(t(expr)),
    file.path(DATA_PATH, paste0(expression_fname, ".feather"))
)

save_feather(
    genes,
    file.path(DATA_PATH, paste0(genes_fname, ".feather"))
)

save_feather(
    circuits,
    file.path(DATA_PATH, paste0(circuits_fname, ".feather"))
)

save_feather(
    as.data.frame(t(fanconi_pathvals)),
    file.path(DATA_PATH, paste0(fanconi_pathvals_fname, ".feather"))
)

save_feather(
    fanconi_circuits,
    file.path(DATA_PATH, paste0(fanconi_circuits_fname, ".feather"))
)
