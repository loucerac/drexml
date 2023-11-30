#!/usr/bin/env bash

mkdir -p results

conda create -y -p ./.venv python=3.10 -c conda-forge

# install latest stable version fo drexml
CONDA_RUN="conda run --live-stream --no-capture-output -p ./.venv"
${CONDA_RUN} pip install drexml

# run drexml using all CPUs and no GPUs
${CONDA_RUN} drexml run --n-gpus 0 experiment.env > results/drexml.out 2> results/drexml.err
rm -rf results/tmp
gzip results/*symbol.tsv
rm -f results/*.tsv

# plot

${CONDA_RUN} drexml plot \
 results/shap_selection_symbol.tsv.gz \
 results/shap_summary_symbol.tsv.gz \
 results/stability_results_symbol.tsv.gz \
 results/
