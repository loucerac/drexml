#!/usr/bin/env bash
BATCH_FILE="hord.sbatch"
DISEASE="fanconi"
MLMODEL="morf"
OPT="hyperopt"
SEED=42
MODE="test"
PATHWAY1="hsa03460m"
PATHWAY2="hsa04110"

export $(egrep -v '^#' .env | xargs)

run_() {

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

    # name, gset, pathway1, pathway2
    OUT_FOLDER="${OUT_PATH}/${DISEASE}/${NAME}/$1/${MLMODEL}/${OPT}/${MODE}/${SEED}"
    mkdir -p OUT_FOLDER
    git archive -o code_snapshot.zip HEAD
    zip -rv code_snapshot.zip .env
    mv code_snapshot.zip "${OUT_FOLDER}/code_snapshot.zip"
    echo "Code snapshot saved to ${OUT_FOLDER}"
    JOB_NAME="hord_${DISEASE}_${MODE}_${NAME}_$1"

    if [[ -z "$2" &&  -z "$3" ]];
    then
        python hord.py --disease ${DISEASE} --mlmodel ${MLMODEL} --opt ${OPT} --seed ${SEED} --mode ${MODE} --gset $1
    elif [[ -z "$3" ]];
    then
        python hord.py --disease ${DISEASE} --mlmodel ${MLMODEL} --opt ${OPT} --seed ${SEED} --mode ${MODE} --pathways $2 --gset $1
    else
        python hord.py --disease ${DISEASE} --mlmodel ${MLMODEL} --opt ${OPT} --seed ${SEED} --mode ${MODE} --pathways $2 --pathways $3 --gset $1
    fi
}

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
