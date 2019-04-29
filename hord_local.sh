#!/usr/bin/env bash
BATCH_FILE="hord.sbatch"
DISEASE="fanconi"
MLMODEL="morf"
OPT="hyperopt"
SEED=42
MODE="test"
PATHWAY1="hsa03460m"
PATHWAY2="hsa04110"
GSET="all"

export $(egrep -v '^#' .env | xargs)

OUT_FOLDER=$(python hord.py --disease ${DISEASE} --mlmodel ${MLMODEL} --opt ${OPT} --seed ${SEED} --mode ${MODE} --gset ${GSET})
OUT_FOLDER=$(echo ${OUT_FOLDER} | awk '{print $NF}')

git archive -o code_snapshot.zip HEAD
zip -rv code_snapshot.zip .env
mv code_snapshot.zip "${OUT_FOLDER}/code_snapshot.zip"

echo "Code snapshot saved to ${OUT_FOLDER}"
