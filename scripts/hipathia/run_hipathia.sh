#!/usr/bin/env 

VERSION="v10"
GTEX_FNAME="GTEx_Analysis_v10_RNASeQCv2.4.2_gene_reads.gct.gz"
GTEX_URL="https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/${GTEX_FNAME}"

mkdir -p data/{rw,interim,final}

[[ -e data/raw/$GTEX_FNAME ]] || wget -P data/raw $GTEX_URL

(cd data/raw && sha256sum -c ${GTEX_FNAME})

mamba env create -y -p ./.venv -f environment.yml

mamba run -p ./.venv Rscript --vanilla run_hipathia.R $GTEX_FNAME $VERSION

