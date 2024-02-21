#!/bin/bash -l

USE_GPU="$1"

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $BASEDIR

declare -a map_size_lst=( 1 25 50 75 100)
for i in "${map_size_lst[@]}"; do
	this_i=$(printf "%03d\n" "$i")
	if [ $USE_GPU == 1 ]; then
		( cd ./experiments/disease_${this_i} \
		&& \
		${CONDA_RUN} hyperfine --show-output --runs 10 --shell=bash --export-csv disease_${this_i}.csv 'drexml run disease.env' )
	else
		sbatch -J drexml-b${this_i} -n 32 --mem 100g -o $this_i.out -e $this_i.err --export=this_i=$this_i,BASEDIR=$BASEDIR hyperfine.sbatch
	fi
done
