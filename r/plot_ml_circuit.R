library("hipathia")
library(stringr)
library(here)

dotenv::load_dot_env(file = here(".env"))
DATA_PATH <- Sys.getenv("DATA_PATH")

use_faconi_metaginfo <- TRUE

if (use_faconi_metaginfo) {
    metaginfo_fname <- ""
    metaginfo_fpath <- file.path(DATA_PATH, metaginfo_fname)
    load(metaginfo_fpath)
} else {
    pathways <- load_pathways("hsa")
}

circuit_ml_name <- "P.hsa03460m.48"
circuit_ml_name_dot_split <- str_split(circuit_ml_name, "[.]", simplify=TRUE)
n_split <- length(circuit_ml_name_dot_split)
pathway_name <- circuit_ml_name_dot_split[[2]]
circuit_name <- str_c(circuit_ml_name_dot_split[1:3], collapse="-")
if (n_split > 3) {
    circuit_name <- str_c(
        c(circuit_name, circuit_ml_name_dot_split[3:n_split]),
        collapse=" "
        )
}

circuit_graph <- pathways[["pathigraphs"]][[pathway_name]][["effector.subgraphs"]][[circuit_name]]

hipathia:::plot_pathigraph(pathways)
