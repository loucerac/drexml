rm(list=ls())

library("hipathia")
library(stringr)
library(here)

dotenv::load_dot_env(file = here(".env"))
DATA_PATH <- Sys.getenv("DATA_PATH")

fanconi_pathvals_fname <- "expreset_pathvals_FA.rds"
pathvals <- as.data.frame(t(readRDS(file.path(DATA_PATH, fanconi_pathvals_fname))))

fanconi_circuits_fname <- "circuits_FA.rds"
fanconi_circuits <- readRDS(file.path(DATA_PATH, fanconi_circuits_fname))
rownames(fanconi_circuits) <- fanconi_circuits$hipathia

query <- names(pathvals) %in% rownames(fanconi_circuits)[fanconi_circuits$in_disease]
pathvals <- pathvals[, query]

use_faconi_metaginfo <- TRUE

if (use_faconi_metaginfo) {
    metaginfo_fname <- "metaginfo_fanconi"
    metaginfo_fpath <- file.path(DATA_PATH, metaginfo_fname)
    # load(metaginfo_fpath)
    pathways <- readRDS(metaginfo_fpath)
} else {
    pathways <- load_pathways("hsa")
}

circuit_ml_name <- "P.hsa03460m.48"
circuit_ml_name_dot_split <- str_split(circuit_ml_name, "[.]", simplify=TRUE)
n_split <- length(circuit_ml_name_dot_split)
pathway_name <- circuit_ml_name_dot_split[[2]]
pathway_name <- str_replace(pathway_name, "m", "_marina")
circuit_name <- str_c(circuit_ml_name_dot_split[1:3], collapse="-")
if (n_split > 3) {
    circuit_name <- str_c(
        c(circuit_name, circuit_ml_name_dot_split[3:n_split]),
        collapse=" "
        )
}

circuit_graph <- pathways[["pathigraphs"]][[pathway_name]][["effector.subgraphs"]][[circuit_name]]

hipathia:::plot_pathigraph(circuit_graph)

summary.list = function(x){
    Quantile<-quantile(x, na.rm=TRUE)
    cv <- ((Quantile[4] - Quantile[2]) / 2.0) / ((Quantile[2] + Quantile[4]) / 2.0)[[1]]
    l = list(
        N.with.NA.removed= length(x[!is.na(x)]),
        Count.of.NA= length(x[is.na(x)]),
        Mean=mean(x, na.rm=TRUE),
        Median=median(x, na.rm=TRUE),
        Max.Min=range(x, na.rm=TRUE),
        Range=max(x, na.rm=TRUE) - min(x, na.rm=TRUE),
        Variance=var(x, na.rm=TRUE),
        Std.Dev=sd(x, na.rm=TRUE),
        Coeff.Variation.Prcnt=sd(x, na.rm=TRUE)/mean(x, na.rm=TRUE)*100,
        Std.Error=sd(x, na.rm=TRUE)/sqrt(length(x[!is.na(x)])),
        Quantile=Quantile,
        cv_norm=cv
    )

    return(l)
}

l <- summary.list(pathvals[[circuit_name]])

print(l)
