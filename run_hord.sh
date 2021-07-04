#!/usr/bin/env bash

# some weird conda init things (cluster and gpu server issues)
source ~/.bashrc
# read and export .env
EXPERIMENT_PATH=$1
export $(egrep -v '^#' .env | xargs)
conda activate ./.venv
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
STARTTIME=$(date +%s)
parentdir="$(dirname "${EXPERIMENT_PATH}")"
parentdir="${parentdir}/ml"
rm -rf ${parentdir}
mkdir -p ${parentdir}
out_path="${parentdir}/out.log"
err_path="${parentdir}/err.log"


python hord.py --n-jobs 24 --gpu --debug --disease ${EXPERIMENT_PATH} > ${out_path} 2> ${err_path}
#python src/explain.py data/experiments/RP_2021/ml/tmp/ 100 1 33 1

ENDTIME=$(date +%s)

t=$(($ENDTIME - $STARTTIME))

echo "End disease ${DISEASE} in ${t} s" >> ${out_path}

