#!/bin/bash
#SBATCH --export=ALL
#SBATCH --mail-user=carlos.loucera@juntadeandalucia.es
#SBATCH --mail-type=ALL
#SBATCH --mem=50g
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -J dreml_24_50
#SBATCH -e dreml_24_50.err
#SBATCH -o dreml_24_50.out

module load anaconda2
conda activate ./.venv
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 

EXPERIMENT_PATH="/home/cloucera/projects/dreml/data/46,XY_complete_gonadal_dysgenesis/experiment.env"
python hord.py --n-jobs 24 --debug --no-gpu --disease ${EXPERIMENT_PATH} 
