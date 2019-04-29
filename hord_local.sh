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

NAME="all"
OUT_FOLDER="${OUT_PATH}/${DISEASE}/${NAME}/${GSET}/${MLMODEL}/${OPT}/${MODE}/${SEED}"
mkdir -p OUT_FOLDER
git archive -o code_snapshot.zip HEAD
zip -rv code_snapshot.zip .env
mv code_snapshot.zip "${OUT_FOLDER}/code_snapshot.zip"
echo "Code snapshot saved to ${OUT_FOLDER}"
JOB_NAME="hord_${DISEASE}_${MODE}_${NAME}_${GSET}"
python hord.py --disease ${DISEASE} --mlmodel ${MLMODEL} --opt ${OPT} --seed ${SEED} --mode ${MODE} --gset ${GSET}
