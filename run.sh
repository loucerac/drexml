#!/usr/bin/env bash

# some weird conda init things (cluster and gpu server issues)
source ~/.bashrc

DATA_PATH="$1"
conda activate ./.venv
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
echo $( which python )

DISEASE_LIST=$(find ${DATA_PATH} -mindepth 1 -maxdepth 1 -type d | sort)
echo $DISEASE_LIST
DISEASES_DONE_FPATH="${DATA_PATH}/diseases_done.txt"
#rm ${DISEASES_DONE_FPATH}
#touch ${DISEASES_DONE_FPATH}

for DISEASE_FOLDER in ${DISEASE_LIST}; do
    echo $DISEASE_FOLDER
    DISEASE=$(basename ${DISEASE_FOLDER})
    is_disease_done=$( grep  "${DISEASE}" "${DISEASES_DONE_FPATH}" )
    if [ -n "${is_disease_done}" ]; then 
        echo "${DISEASE} already done"
    else
        echo "Begin disease ${DISEASE}"
        #DISEASE_FOLDER="${DATA_PATH}/results/${DISEASE}"
        EXPERIMENT_PATH="${DISEASE_FOLDER}/experiment.env"
        echo "Experiment $EXPERIMENT_PATH"
        ML_FOLDER="${DISEASE_FOLDER}/ml"
        TMP_FOLDER="${ML_FOLDER}/tmp"
        rm -rf "${DISEASE_FOLDER}/ml"
        mkdir -p "${TMP_FOLDER}"
        out_path="${ML_FOLDER}/out.log"
        err_path="${ML_FOLDER}/err.log"

        STARTTIME=$(date +%s)
        
        python hord.py --n-jobs 110 --gpu --disease ${EXPERIMENT_PATH} > ${out_path} 2> ${err_path}

        ENDTIME=$(date +%s)

        t=$(($ENDTIME - $STARTTIME))

        echo "End disease ${DISEASE} in ${t} s"
        echo "End disease ${DISEASE} in ${t} s" >> ${out_path}
        echo "End disease ${DISEASE} in ${t} s" >> ${DISEASES_DONE_FPATH}
        sleep 1m
    fi
done
