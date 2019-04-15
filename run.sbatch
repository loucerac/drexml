#!/bin/bash 
#SBATCH --export=ALL
#SBATCH --mail-user=carlos.loucera@juntadeandalucia.es
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem=400000
#SBATCH -o jobs/%A_%a.out # Standard output
#SBATCH -e jobs/%A_%a.err # Standard error

source ~/.bashrc
conda activate hord

python rf.py
python relevance_v2.py
