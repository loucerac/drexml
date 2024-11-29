#!/usr/bin/env bash

DRUGBANK_VERSION="5-1-12"
MYGENE_VERSION=$(date +%Y%m%d)

# # Download drugbank
#pixi run download_drugbank --version $DRUGBANK_VERSION --filename data/raw/drugbank_v$DRUGBANK_VERSION.zip

# # Check integrity
#pixi run check_drugbank

# # Parse drugbank
#pixi run parse_drugbank data/raw/drugbank_v$DRUGBANK_VERSION.zip data/final/drugbank_v$DRUGBANK_VERSION.tsv.gz

# Translate drugbank
pixi run translate_drugbank data/final/drugbank_v$DRUGBANK_VERSION.tsv.gz data/final/genes_drugbank-${DRUGBANK_VERSION}_mygene-${MYGENE_VERSION}.tsv.gz
