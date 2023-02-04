#### GET GENES AND PATHWAYS FOR MACHINE LEARNING ####

if (!require(pacman)) {
  install.packages("pacman")
  library(pacman)
}else{ library(pacman)}

p_load("magrittr", "here", "hipathia", "AnnotationDbi", "org.Hs.eg.db", "data.table", "VarfromPDB", "openxlsx", "feather")

#### 1. Get disease ORPHA-genes  ####

localPDB(localPDB.path = paste(getwd(), "data", "interim","localPDB", sep = "/"), PDB = "all",
         omim.url = "ftp://ftp.ncbi.nih.gov/gene/DATA/gene2refseq.gz", download.method = "curl_fetch_disk")

disease = "Retinitis pigmentosa"
short_dis = "RP"
genes_HPO_disease<- pheno_extract_HPO(disease, localPDB.path = paste(getwd(), "data", "interim","localPDB", sep = "/"))
table_genes <- genes_HPO_disease[!duplicated(genes_HPO_disease[,"GeneName"]), ] %>% subset(., .$GeneName!="") %>% .[, -5]

if(!dir.exists("./results/tables")){
  dir.create("./results/tables")
}

write.xlsx(table_genes, file = here( "results", "tables",paste0("genes_", disease,".xlsx")), row.names = F )

genes_ORPHA_disease <- as.character(unique(subset(genes_HPO_disease$GeneName, (genes_HPO_disease$GeneName!=""))))
entrezIDgenes_ORPHA <- data.frame(genename = genes_ORPHA_disease, 
                                  entrezID= mapIds(org.Hs.eg.db, keys=genes_ORPHA_disease, column="ENTREZID", keytype="SYMBOL", multiVals="first"),
                                  stringsAsFactors = F) %>% subset(., .$entrezID!="NA")


#### 2.Get all HiPathia Genes ####
# get a list of genes by pathway from pathigraphs object

metaginfo <- hipathia::load_pathways("hsa")
path_list <- read.table(file = here("data","raw", "physiological_paths.tsv"), sep = "\t") ## Load physiological pathways to filter them
pathways <- hipathia:::filter_pathways(metaginfo, pathways_list = as.character(path_list[,2])) ## Filter only physiological pathways


# iterate through pathigraph subgraphs getting all the genesList (entrez)

entrez.list <- lapply(pathways$pathigraphs, function(x) {unlist(get.vertex.attribute(x$graph, name = "genesList"))}) %>% 
  lapply(., function(x){ x[! (is.na(x) | x == "/" | x == "NA")]}) # clean from NA, "", "/"

# unlist and clean from NA , /
entrezs <- as.character(unlist(entrez.list))
entrezs <- unique(entrezs[! (is.na(entrezs) | entrezs == "/" | entrezs == "NA")])



#### 3. Get ORPHA genes in Hipathia and circuits containning them ####

# Get orpha genes (from studied disease) in Hipathia
orpha_hi <- as.data.frame(entrezs)
orpha_hi <- cbind(orpha_hi, entrezs%in%entrezIDgenes_ORPHA$entrezID )
colnames(orpha_hi) <- c("genes_hipathia","orphagenes_in_hip")

# Get all HiPathia circuits
subpathways.list <- lapply(pathways$pathigraphs, function(x){names(x$effector.subgraphs)})
subpathways <- as.data.frame(unlist(subpathways.list))
colnames(subpathways) <- "hipathia"

# Table of all Hipathia pathways with orpha_genes
cualpath_dis<-function(genes, pathway){
  
  w<-intersect(genes, entrez.list[[pathway]])
  
  if(length(w) != 0){
    return(T)
  }else{return(F)}
  
}

paths_disease <-sapply( names(entrez.list), function(x){cualpath_dis(as.character(entrezIDgenes_ORPHA$entrezID),x)})

paths_disease <- names(entrez.list)[paths_disease]

dis_circuits <- subpathways.list[paths_disease]

write.xlsx(paths_disease, file = here("results","tables",paste0("pathways_", short_dis ,".xls")), row.names = F, col.names = T)


## Get circuits with orpha_genes

cir_disease <-as.character(unlist(dis_circuits))

## Function to get the entrez of a given circuit ##
get_entrez_cir <- function(subpath_name){
  
  path_name <- str_extract(subpath_name, "hsa[0-9]+")
  circuit <- pathways[["pathigraphs"]][[path_name]][["effector.subgraphs"]][[subpath_name]]
  entrez <-V(pathways[["pathigraphs"]][[path_name]][["effector.subgraphs"]][[subpath_name]])$genesList %>% unlist(.)
  idx<- which(entrez=="NA" | is.na(entrez) | entrez== "/")
  
  if(length(idx)!= 0){
    
    idx<- which(entrez=="NA" | is.na(entrez) | entrez== "/")
    entrez <- entrez[-idx]
  }
  
  return(entrez)
  
}

cir_disease_entrez <- sapply(cir_disease, function(x){get_entrez_cir(x)})

## Function to to extract the circuits where entrex_disease genes are located
which_cir <- function(entrez_disease,circuit){
  
  w <-intersect(entrez_disease, circuit)
  
  if(length(w) != 0){
    return(T)
  }else{return(F)}
  
}

af_cir <- lapply(cir_disease_entrez,function(x){which_cir(entrezIDgenes_ORPHA$entrezID,x)})
af_cir <-stack(af_cir) 

cir_af <- data.frame(Circuit = get_path_names(metaginfo, names = as.character(af_cir$ind[af_cir$values==T])),
                     Hipathia_code = af_cir$ind[af_cir$values==T], stringsAsFactors = F)

write.xlsx(cir_af, file = here("results", "tables",paste0("circuits_ORPHA",short_dis ,".xls")), row.names = F, col.names = T)

#### 4. HPO- disease related genes ####
## Load table of HPO-orpha disesases- genes from HPOdb.obo and phenotype files tagged with levels and genes  (obtained in scrip HPO_obo_annotations.R)

HPOdb <- readRDS(here("data", "interim", "hpo_or_p_entrezs05122019.rds"))

## Load phenotype_to_genes.txt file from HPOdb (downloaded in local)

hpo2genes <- data.frame(read.delim(skip = 1, file = (here("data", "raw", "phenotype_to_genes20191010.txt")),fill = T, header = F, sep="\t"), stringsAsFactors = T)
colnames(hpo2genes) <- c("hpo_id", "term_id", "entrez", "symbol")

hpo_genes_long <-  data.frame(aggregate(cbind(hpo2genes$entrez) ~ as.character(hpo2genes$hpo_id), data = hpo2genes , FUN = paste ), stringsAsFactors = F)
colnames(hpo_genes_long)<- c("HPO","entrez_id")

genes_hpo_long <- data.frame(aggregate(cbind(as.character(hpo2genes$hpo_id)) ~ hpo2genes$entrez, data = hpo2genes , FUN = paste ), stringsAsFactors = F)
colnames(genes_hpo_long)<- c("entrez_id","HPO")

## Select HPO codes related to the ORPHA disease

orphacode_RD <- "ORPHA:791" ## RP Orphanet's ID

hpos_RD <- HPOdb[HPOdb$disease_resource == orphacode_RD,]

hpo_codes <- hpos_RD$hpo_id [hpos_RD$level>=7] ## Select HPO>= 7 of specificity

write.xlsx(hpos_RD[hpos_RD$level >= 7, c(2,8,9) ], file = here("results","tables","hpos_RP.xlsx"), row.names = F)

## Get entrezs with all RD HPOs as tag (22)

indexall_entrezs <- sapply(genes_hpo_long$HPO,function(x){ all(hpo_codes %in% x)})
entrezs_allRDHPO <- as.character(genes_hpo_long$entrez_id[indexall_entrezs])

sum(entrezs_allRDHPO %in% metaginfo$all.genes)

df_allRDHPO <- data.frame(Gene_symbol = mapIds(org.Hs.eg.db, keys = entrezs_allRDHPO, column = "SYMBOL", keytype = "ENTREZID"), Entrez = entrezs_allRDHPO)

write.xlsx(df_allRDHPO, file = here("results","tables","genes_22RDHPO.xls"), row.names = F)

## Get entrezs with 14/22 RD HPOs as tag 
which_cir14 <- function(entrez_disease,circuit){
  
  w <-intersect(entrez_disease, circuit)
  
  if(length(w) >= 14){
    return(T)
  }else{return(F)}
  
}

index14_entrezs <- sapply(genes_hpo_long$HPO, function(x){ which_cir14( hpo_codes, x )})
entrezs_14RDHPO <- as.character(genes_hpo_long$entrez_id [ index14_entrezs ])

sum(entrezs_14RDHPO %in% pathways$all.genes)


## Get entrezs with 12/22 RD HPOs as tag
which_cir12 <- function(entrez_disease,circuit){
  
  w <-intersect(entrez_disease, circuit)
  
  if(length(w) >= 12){
    return(T)
  }else{return(F)}
  
}

index12_entrezs <- sapply(genes_hpo_long$HPO, function(x){ which_cir12( hpo_codes, x )})
entrezs_12RDHPO <- as.character(genes_hpo_long$entrez_id [ index12_entrezs ])

sum(entrezs_12RDHPO %in% pathways$all.genes)




## Get entrezs with 10/22 RD HPOs as tag
which_cir10 <- function(entrez_disease,circuit){
  
  w <-intersect(entrez_disease, circuit)
  
  if(length(w) >= 10){
    return(T)
  }else{return(F)}
  
}

index10_entrezs <- sapply(genes_hpo_long$HPO, function(x){ which_cir10( hpo_codes, x )})
entrezs_10RDHPO <- as.character(genes_hpo_long$entrez_id [ index10_entrezs ])

sum(entrezs_10RDHPO %in% pathways$all.genes)

df_10RDHPO <- data.frame(Gene_symbol = mapIds(org.Hs.eg.db, keys = entrezs_10RDHPO, column = "SYMBOL", keytype = "ENTREZID"), Entrez = entrezs_10RDHPO)
write.xlsx(df_10RDHPO, file = here("results","tables","genes_10RDHPO.xls"), row.names = F)


## Get entrezs with 8/22 RD HPOs as tag
which_cir8 <- function(entrez_disease,circuit){
  
  w <-intersect(entrez_disease, circuit)
  
  if(length(w) >= 8){
    return(T)
  }else{return(F)}
  
}

index8_entrezs <- sapply(genes_hpo_long$HPO, function(x){ which_cir8( hpo_codes, x )})
entrezs_8RDHPO <- as.character(genes_hpo_long$entrez_id [ index8_entrezs ])

sum(entrezs_8RDHPO %in% pathways$all.genes)

df_8RDHPO <- data.frame(Gene_symbol = mapIds(org.Hs.eg.db, keys = entrezs_8RDHPO, column = "SYMBOL", keytype = "ENTREZID"), Entrez = entrezs_8RDHPO)
write.xlsx(df_8RDHPO, file = here("results","tables","genes_8RDHPO.xls"), row.names = F)


## Add ORPHA RD entrezs with HPO related entrez (Select as filter genes that share 10/22 HPOs with the disease RP) ###

RD_entrezs <- unique(c(entrezIDgenes_ORPHA$entrezID, entrezs_10RDHPO)) 

# Get pathways with RD genes
paths_RD <- sapply(names(entrez.list), function(x){cualpath_dis(as.character(RD_entrezs),x)})

paths_RD <- names(entrez.list)[paths_RD]

write.xlsx(path_list[as.character(path_list$V2) %in% paths_RD, ], file = here("results","tables","paths_RD.xls"), row.names = F)

# Get circuits from pathways with RD genes.   

cir_RD <- subpathways.list[paths_RD] %>% unlist(.) %>% as.vector(.)

cir_RD_entrez <- sapply(cir_RD, function(x){get_entrez_cir(x)})


RD_cir <- lapply(cir_RD_entrez,function(x){which_cir(RD_entrezs,x)})
RD_cir <-stack(RD_cir) 



cir_RD_names <- data.frame(Circuit = get_path_names(metaginfo, names = as.character(RD_cir$ind[RD_cir$values==T])),
                           Hipathia_code = RD_cir$ind[RD_cir$values==T], stringsAsFactors = F)

write.xlsx(cir_RD_names, file = here("results", "tables",paste0("cirHPO",disease,".xlsx")),row.names = F)

## Table of All Hipathia circuits and disease affected circuits (ORPHA+HPO)
circuits <- cbind(subpathways, subpathways$hipathia %in% RD_cir$ind[RD_cir$values==T])
colnames(circuits) <- c("hipathia", "in_disease")

saveRDS(circuits, file = here("data","final","circuits_RP.rds"))


## Save files into ".feather"
save_feather <- function(x, path) {
  df <- data.frame(index = row.names(x), x)
  
  feather::write_feather(df, path)
}


save_feather(
  circuits,
  file.path(here("data","final",paste0("circuits_",short_dis,".rds.feather")))
)
