#!/usr/bin/env bash
BATCH_FILE="hord.sbatch"
DISEASE="fanconi"
MLMODEL="morf"
OPT="hyperopt"
SEED=42
MODE="train"
PATHWAY1="hsa03460m"
PATHWAY2="hsa04110"

# Read .env file
export $(egrep -v '^#' .env | xargs)


run_() {
    # name, gset, pathway1, pathway2

    if [[ -z "$2" &&  -z "$3" ]];
    then
        NAME="all"
    elif [ -z "$3" ];
    then
        NAME=$2
    else
        NAME="$2_$3"
    fi

    echo "${NAME}"

    OUT_FOLDER="${OUT_PATH}/${DISEASE}/${NAME}/$1/${MLMODEL}/${OPT}/${MODE}/${SEED}"
    mkdir -p ${OUT_FOLDER}

    git archive -o code_snapshot.tar.gz HEAD
    gunzip -f code_snapshot.tar.gz
    tar -uf code_snapshot.tar .env
    gzip code_snapshot.tar
    mv code_snapshot.tar.gz "${OUT_FOLDER}/code_snapshot.tar.gz"
    echo "Code snapshot saved to ${OUT_FOLDER}"
    JOB_NAME="hord_${DISEASE}_${MODE}_${NAME}_$1"

    if [[ -z "$2" &&  -z "$3" ]];
    then
        sbatch -J ${JOB_NAME} --export=DISEASE=${DISEASE},MLMODEL=${MLMODEL},OPT=${OPT},SEED=${SEED},MODE=${MODE},GSET=$1 ${BATCH_FILE}
    elif [[ -z "$3" ]];
    then
        sbatch -J ${JOB_NAME} --export=DISEASE=${DISEASE},MLMODEL=${MLMODEL},OPT=${OPT},SEED=${SEED},MODE=${MODE},PATHWAY1=$2,GSET=$1 ${BATCH_FILE}
    else
        sbatch -J ${JOB_NAME} --export=DISEASE=${DISEASE},MLMODEL=${MLMODEL},OPT=${OPT},SEED=${SEED},MODE=${MODE},PATHWAY1=$2,PATHWAY2=$3,GSET=$1 ${BATCH_FILE}
    fi
}


####################
## All genes
####################

GSET="all"
run_ $GSET

GSET="all"
run_ $GSET $PATHWAY1

GSET="all"
run_ $GSET $PATHWAY1 $PATHWAY2

####################
## Target genes
####################

GSET="target"
run_ $GSET

GSET="target"
run_ $GSET $PATHWAY1

GSET="target"
run_ $GSET $PATHWAY1 $PATHWAY2
