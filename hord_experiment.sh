#!/usr/bin/env bash
BATCH_FILE="hord_experiment.sbatch"
MLMODEL="morf"
OPT="hyperopt"
SEED=42
LOCAL=0

MODE=$2
EXPERIMENT_ENV_FILE=$1
EXPERIMENT_DIR="$(dirname "${EXPERIMENT_ENV_FILE}")"
EXPERIMENT="$(basename "$EXPERIMENT_DIR")"
parentdir="$(dirname "${EXPERIMENT_DIR}")"
DISEASE="$(basename "$parentdir")"

OUT_FOLDER="${EXPERIMENT_DIR}/ml/${MLMODEL}_${MODE}"

# Read .env file
export $(egrep -v '^#' .env | xargs)

mkdir -p ${OUT_FOLDER}

git archive -o code_snapshot.tar.gz HEAD
gunzip -f code_snapshot.tar.gz
tar -uf code_snapshot.tar .env
gzip code_snapshot.tar
mv code_snapshot.tar.gz "${OUT_FOLDER}/code_snapshot.tar.gz"
echo "Code snapshot saved to ${OUT_FOLDER}"
JOB_NAME="hord_${DISEASE}_${EXPERIMENT}_${MODE}"
ERR_FILE="${OUT_FOLDER}/${JOB_NAME}.err"
OUT_FILE="${OUT_FOLDER}/${JOB_NAME}.out"

if [ $LOCAL -eq 1 ] ; then
    # local
    python hord.py --disease ${EXPERIMENT_ENV_FILE} --mlmodel ${MLMODEL} --opt ${OPT} --seed ${SEED} --mode ${MODE}
else
    sbatch -J ${JOB_NAME} -n ${NUM_CPUSqu} -e ${ERR_FILE} -o ${OUT_FILE} --export=EXPERIMENT_ENV_FILE=${EXPERIMENT_ENV_FILE},MLMODEL=${MLMODEL},OPT=${OPT},SEED=${SEED},MODE=${MODE} ${BATCH_FILE}
fi
