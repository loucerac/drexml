#!/usr/bin/env bash

UPDATE_GENE_INFO=false
if [ "$UPDATE_GENE_INFO" = true ] ; then
    MYGENE_VERSION=$( date +%Y%m%d )
    echo "Update Mygene version to ${MYGENE_VERSION}"
else
    MYGENE_VERSION="20230120"
    echo "Using Mygene version ${MYGENE_VERSION}"
fi
GTEX_VERSION="V8"
DRUGBANK_VERSION="v050108"

THIS_FOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_PATH="$THIS_FOLDER/data"
RAW_FOLDER="${DATA_PATH}/raw"
INTERIM_FOLDER="${DATA_PATH}/interim"
FINAL_FOLDER="${DATA_PATH}/final"

# RAW
XML_PATH="${RAW_FOLDER}/drugbank-${DRUGBANK_VERSION}.xml.gz"
PARQUET_PATH="${FINAL_FOLDER}/expreset_Hinorm_gtex${GTEX_VERSION}.rds.feather"

# INTERIM
TSV_PATH="${INTERIM_FOLDER}/drugbank-${DRUGBANK_VERSION}.tsv"
DB_GENES_PATH="${INTERIM_FOLDER}/genes_drugbank-${DRUGBANK_VERSION}_mygene-${MYGENE_VERSION}.tsv"
GTEX_GENES_PATH="${INTERIM_FOLDER}/genes_gtex-${GTEX_VERSION}_mygene-${MYGENE_VERSION}.tsv"

# FINAL
DB_FILTERED_PATH="${FINAL_FOLDER}/drugbank-${DRUGBANK_VERSION}_gtex-${GTEX_VERSION}_mygene-${MYGENE_VERSION}.tsv"
GENES_FILTERED_PATH="${FINAL_FOLDER}/genes-drugbank-${DRUGBANK_VERSION}_gtex-${GTEX_VERSION}_mygene-${MYGENE_VERSION}.tsv"

PARSER_FOLDER="${THIS_FOLDER}/modules/drugbank-parser"

# Parsing
python ${PARSER_FOLDER}/parser.py parse $XML_PATH $TSV_PATH

if test -f "$DB_GENES_PATH"; then
    echo "$DB_GENES_PATH exists."
else
    echo "$DB_GENES_PATH does no exist."
    if [ "$UPDATE_GENE_INFO" = true ] ; then
        echo "Updating Mygene"
        python ${PARSER_FOLDER}/parser.py translate $TSV_PATH $DB_GENES_PATH --kind drugbank 
        python ${PARSER_FOLDER}/parser.py translate $PARQUET_PATH $GTEX_GENES_PATH --kind gtex 
    else
        echo "Either search for the corresponding Mygene $MYGENE_VERSION$ file or set UPDATE to true. Then, launch again the script."
    fi
fi

python ${PARSER_FOLDER}/parser.py filter \
    --drugbank-path ${TSV_PATH} \
    --drugbank-genes-path ${DB_GENES_PATH} \
    --gtex-genes-path ${GTEX_GENES_PATH} \
    ${DB_FILTERED_PATH} \
    ${GENES_FILTERED_PATH}
