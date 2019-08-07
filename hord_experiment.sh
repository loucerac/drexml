#!/usr/bin/env bash
BATCH_FILE="hord_experiment.sbatch"
MLMODEL="morf"
OPT="hyperopt"
SEED=42
MODE="train"

EXPERIMENT_ENV_FILE=$1
EXPERIMENT_DIR="$(dirname "${EXPERIMENT_ENV_FILE}")"
EXPERIMENT="$(basename "$EXPERIMENT_DIR")"
parentdir="$(dirname "${EXPERIMENT_DIR}")"
DISEASE="$(basename "$parentdir")"

OUT_FOLDER="${EXPERIMENT_DIR}/ml/${MLMODEL}"

# Read .env file
export $(egrep -v '^#' .env | xargs)

mkdir -p ${OUT_FOLDER}

git archive -o code_snapshot.tar.gz HEAD
gunzip -f code_snapshot.tar.gz
tar -uf code_snapshot.tar .env
gzip code_snapshot.tar
mv code_snapshot.tar.gz "${OUT_FOLDER}/code_snapshot.tar.gz"
echo "Code snapshot saved to ${OUT_FOLDER}"
JOB_NAME="hord_${DISEASE}_${EXPERIMENT}"

sbatch -J ${JOB_NAME} --export=EXPERIMENT_ENV_FILE=${EXPERIMENT_ENV_FILE},MLMODEL=${MLMODEL},OPT=${OPT},SEED=${SEED},MODE=${MODE} ${BATCH_FILE}
