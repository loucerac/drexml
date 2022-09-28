#!/bin/bash
#SBATCH --export=ALL
#SBATCH --mail-user=carlos.loucera@juntadeandalucia.es
#SBATCH --mail-type=ALL
#SBATCH --mem=10G
#SBATCH -c 1


module load anaconda2
conda activate ./.venv

python drexml/ml_plots.py $MLPATH 0 0
