#!/usr/bin/env bash

# some weird conda init things (cluster and gpu server issues)
source ~/.bashrc
# read and export .env

export $(egrep -v '^#' .env | xargs)
conda activate ./.venv
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 

DISEASE_LIST=$(find ${DATA_PATH}/results  -mindepth 1 -maxdepth 1 -type d ! -path ${DATA_PATH}/results | sort)
DISEASES_DONE_FPATH="${DATA_PATH}/results/diseases_done.txt"
#rm ${DISEASES_DONE_FPATH}
#touch ${DISEASES_DONE_FPATH}

for f in ${DISEASE_LIST}; do
    echo $f
    DISEASE=$(basename ${f})
    is_disease_done=$( grep  "${DISEASE}" "${DISEASES_DONE_FPATH}" )
    if [ -n "${is_disease_done}" ]; then 
        echo "${DISEASE} already done"
    else
        echo "Begin disease ${DISEASE}"
        DISEASE_FOLDER="${DATA_PATH}/results/${DISEASE}"
        EXPERIMENT_PATH="${DISEASE_FOLDER}/experiment.env"
        TMP_FOLDER="${DISEASE_FOLDER}/ml/tmp"
        rm -rf "${DISEASE_FOLDER}/ml"
        mkdir "${DISEASE_FOLDER}/ml"

        STARTTIME=$(date +%s)
        
        python hord.py --n-jobs 24 --gpu --disease ${EXPERIMENT_PATH} 

        ENDTIME=$(date +%s)

        t=$(($ENDTIME - $STARTTIME))

        echo "End disease ${DISEASE} in ${t} s"

        echo "End disease ${DISEASE} in ${t} s" >> ${DISEASES_DONE_FPATH}
        sleep 5m
    fi
done
