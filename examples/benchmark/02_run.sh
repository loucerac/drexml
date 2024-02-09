#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $BASEDIR
set -a            
source .env
set +a

conda deactivate

# install latest stable version of drexml
CONDA_RUN="conda run --live-stream --no-capture-output -p ${BASEDIR}/.venv"

declare -a map_size_lst=( 1 25 50 75 100)
for i in "${map_size_lst[@]}"; do
	this_i=$(printf "%03d\n" "$i")
	cd $BASEDIR/experiments/disease_${this_i}
	sbatch -J drexml-b${this_i} -n 50 --mem 100g  --wrap="${CONDA_RUN} hyperfine --runs 10 --shell=bash --export-csv disease_${this_i}.csv 'drexml run disease.env'" -o log.out -e log.err
done
