#!/usr/bin/env bash
BATCH_FILE="hord.sbatch"
DISEASE="fanconi"
MLMODEL="morf"
OPT="hyperopt"
SEED=42
MODE="train"
PATHWAY1="hsa03460m"
PATHWAY2="hsa04110"
GSET="all"

# Read .env file
export $(egrep -v '^#' .env | xargs)

NAME="all"
OUT_FOLDER="${OUT_PATH}/${DISEASE}/${NAME}/${GSET}${MLMODEL}/${OPT}/${MODE}/${SEED}"
mkdir -p ${OUT_FOLDER}
git archive -o code_snapshot.tar.gz HEAD
gunzip -f code_snapshot.tar.gz
tar -uf code_snapshot.tar .env
gzip code_snapshot.tar
mv code_snapshot.tar.gz "${OUT_FOLDER}/code_snapshot.tar.gz"
echo "Code snapshot saved to ${OUT_FOLDER}"
JOB_NAME="hord_${DISEASE}_${MODE}_${NAME}_${GSET}"
sbatch -J ${JOB_NAME} --export=DISEASE=${DISEASE},MLMODEL=${MLMODEL},OPT=${OPT},SEED=${SEED},MODE=${MODE},GSET=${GSET} ${BATCH_FILE}

NAME="${PATHWAY1}_${PATHWAY2}"
OUT_FOLDER="${OUT_PATH}/${DISEASE}/${NAME}/${GSET}/${MLMODEL}/${OPT}/${MODE}/${SEED}"
mkdir -p ${OUT_FOLDER}
git archive -o code_snapshot.tar.gz HEAD
gunzip -f code_snapshot.tar.gz
tar -uf code_snapshot.tar .env
gzip code_snapshot.tar
mv code_snapshot.tar.gz "${OUT_FOLDER}/code_snapshot.tar.gz"
echo "Code snapshot saved to ${OUT_FOLDER}"
JOB_NAME="hord_${DISEASE}_${MODE}_${PATHWAY1}_${PATHWAY2}_${GSET}"
sbatch -J ${JOB_NAME} --export=DISEASE=${DISEASE},MLMODEL=${MLMODEL},OPT=${OPT},SEED=${SEED},MODE=${MODE},PATHWAY1=${PATHWAY1},PATHWAY2=${PATHWAY2},GSET=${GSET} ${BATCH_FILE}

NAME="${PATHWAY1}"
OUT_FOLDER="${OUT_PATH}/${DISEASE}/${NAME}/${GSET}/${MLMODEL}/${OPT}/${MODE}/${SEED}"
mkdir -p ${OUT_FOLDER}
git archive -o code_snapshot.tar.gz HEAD
gunzip -f code_snapshot.tar.gz
tar -uf code_snapshot.tar .env
gzip code_snapshot.tar
mv code_snapshot.tar.gz "${OUT_FOLDER}/code_snapshot.tar.gz"
echo "Code snapshot saved to ${OUT_FOLDER}"
JOB_NAME="hord_${DISEASE}_${MODE}_${PATHWAY1}_${GSET}"
sbatch -J ${JOB_NAME} --export=DISEASE=${DISEASE},MLMODEL=${MLMODEL},OPT=${OPT},SEED=${SEED},MODE=${MODE},PATHWAY1=${PATHWAY1},GSET=${GSET} ${BATCH_FILE}
