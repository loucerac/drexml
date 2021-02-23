#!/usr/bin/env bash

# some weird conda init things (cluster and gpu server issues)
source ~/.bashrc
# read and export .env
EXPERIMENT_PATH=$1
export $(egrep -v '^#' .env | xargs)
conda activate ./.venv
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 

STARTTIME=$(date +%s)

python hord.py --n-jobs 24 --gpu --disease ${EXPERIMENT_PATH} 

ENDTIME=$(date +%s)

t=$(($ENDTIME - $STARTTIME))

echo "End disease ${DISEASE} in ${t} s"

