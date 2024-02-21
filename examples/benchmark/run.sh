#!/usr/bin/env bash

USE_GPU="$1"

bash 01_insall.sh $USE_GPU
CONDA_RUN="conda run --live-stream --no-capture-output -p ${BASEDIR}/.venv"

$CONDA_RUN python 02_build_benchmark.py
bash 03_run_benchmark.sh $USE_GPU

$CONDA_RUN 04_gather_results.py
