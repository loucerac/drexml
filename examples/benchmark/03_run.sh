#!/bin/bash -l

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $BASEDIR

declare -a map_size_lst=( 1 25 50 75 100)
for i in "${map_size_lst[@]}"; do
	this_i=$(printf "%03d\n" "$i")
	sbatch -J drexml-b${this_i} -n 32 --mem 200g -o $this_i.out -e $this_i.err --export=this_i=$this_i,BASEDIR=$BASEDIR run.sbatch
done
