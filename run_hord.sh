#!/usr/bin/env bash

# some weird conda init things (cluster and gpu server issues)
source ~/.bashrc
# read and export .env
EXPERIMENT_PATH=$1
export $(egrep -v '^#' .env | xargs)
conda activate ./.venv
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=1
STARTTIME=$(date +%s)

python hord.py --n-jobs 33 --gpu --disease ${EXPERIMENT_PATH} 
#python src/explain.py data/experiments/RP_2021/ml/tmp/ 100 1 33 1

ENDTIME=$(date +%s)

t=$(($ENDTIME - $STARTTIME))

echo "End disease ${DISEASE} in ${t} s"

