BATCH_FILE="hord.sbatch"
DISEASE="fanconi"
MLMODEL="morf"
OPT="hyperopt"
SEED=42
MODE="train"
PATHWAY1="hsa03460m"
PATHWAY2="hsa04110"

JOB_NAME="hord_${DISEASE}_${MODE}_1"
sbatch -J ${JOB_NAME} --export=DISEASE=${DISEASE},MLMODEL=${MLMODEL},OPT=${OPT},SEED=${SEED},MODE=${MODE},PATHWAY1=${PATHWAY1},PATHWAY2=${PATHWAY2} ${BATCH_FILE}
JOB_NAME="hord_${DISEASE}_${MODE}_2"
sbatch -J ${JOB_NAME} --export=DISEASE=${DISEASE},MLMODEL=${MLMODEL},OPT=${OPT},SEED=${SEED},MODE=${MODE},PATHWAY1=${PATHWAY1} ${BATCH_FILE}
