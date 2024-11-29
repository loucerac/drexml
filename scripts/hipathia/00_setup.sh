#!/usr/bin/env 

mkdir -p data/{rw,interim,final}

mamba env create -y -p ./.venv -f environment.yml

mamba run -p ./.venv Rscript --vanilla run_hipathia.R $GTEX_FNAME $VERSION
