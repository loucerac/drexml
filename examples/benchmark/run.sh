#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $BASEDIR
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

# install latest stable version of drexml
CONDA_RUN="conda run --live-stream --no-capture-output -p ${BASEDIR}/.venv"
${CONDA_RUN} pip install -I --force-reinstall --no-cache-dir --no-binary=shap drexml==1.0.4

if [ $USE_GPU == 1 ]; then
	${CONDA_RUN} python -c 'import shap; shap.utils.assert_import("cext_gpu")'
fi

${CONDA_RUN} python 01_build_benchmark.py

declare -a map_size_lst=( 1 25 50 75 100)
for i in "${map_size_lst[@]}"; do
	this_i=$(printf "%03d\n" "$i")
	( cd ./experiments/disease_${this_i} \
	&& \
	${CONDA_RUN} hyperfine --runs 10 --shell=bash --export-csv disease_${this_i}.csv 'drexml run disease.env' )
done
