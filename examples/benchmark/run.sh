#!/usr/bin/env bash

set -a            
source .env
set +a

conda deactivate

mkdir -p results

rm -rf ./.venv

if [ $USE_GPU == 1 ]; then
	conda create -y -p ./.venv --override-channels -c "nvidia/label/cuda-11.8.0" \
	-c conda-forge \
	cuda cuda-nvcc cuda-toolkit gxx=11.2 python=3.10 python-dotenv jupyterlab hyperfine
else
	conda create -y -p ./.venv --override-channels -c conda-forge \
	python=3.10 python-dotenv jupyterlab hyperfine
fi

# install latest stable version fo drexml
CONDA_RUN="conda run --live-stream --no-capture-output -p ./.venv"
${CONDA_RUN} pip install -I --force-reinstall --no-cache-dir --no-binary=shap drexml==1.0.4

if [ $USE_GPU == 1 ]; then
	${CONDA_RUN} python -c 'import shap; shap.utils.assert_import("cext_gpu")'
fi

${CONDA_RUN} python 01_build_benchmark.py

${CONDA_RUN} hyperfine --runs 5 --shell=bash --export-csv disease_001.csv 'drexml run ./examples/disease_001/disease.env'

# run drexml using all CPUs and no GPUs
#${CONDA_RUN} drexml run --n-gpus 0 experiment.env > results/drexml.out 2> results/drexml.err
#rm -rf results/tmp
