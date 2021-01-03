#!/usr/bin/env bash
source ~/.bashrc
conda activate ./.venv
DATA="$(pwd)/notebooks/data/PanRD_ML/results"
DISEASE="Acute_promyelocytic_leukemia"
DISEASE_FOLDER="${DATA}/${DISEASE}"
EXPERIMENT_PATH="${DISEASE_FOLDER}/experiment.env"
TMP_FOLDER="${DISEASE_FOLDER}/ml/tmp"
rm -rf "${DISESE_FOLDER}/ml"

PYTHONPATH="${PYTHONPATH}:$(pwd)" python hord.py --gpu --disease ${EXPERIMENT_PATH} "${DISESE_FOLDER}/ml/out.txt" 2> "${DISESE_FOLDER}/ml/err.txt" 

