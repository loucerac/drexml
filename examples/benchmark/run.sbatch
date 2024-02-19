#!/usr/bin/env bash

. '/home/cloucera/miniconda3/etc/profile.d/conda.sh'

CONDA_RUN="conda run --live-stream --no-capture-output -p ${BASEDIR}/.venv"

cd $BASEDIR/experiments/disease_${this_i}
$CONDA_RUN hyperfine --show-output --runs 10 --shell=bash --export-csv disease_${this_i}.csv 'drexml run --no-add disease.env'

