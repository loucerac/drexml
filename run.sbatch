#!/bin/bash 
#SBATCH --export=ALL
#SBATCH --mail-user=carlos.loucera@juntadeandalucia.es
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem=0
#SBATCH -o jobs/%A_%a.out # Standard output
#SBATCH -e jobs/%A_%a.err # Standard error


module load anaconda2
conda activate holrd

python hord.py --disease fanconi --mlmodel morf --opt hyperopt --seed 42 --mode test --pathways hsa03460m --pathways hsa04110  

# source ~/.bashrc
# conda activate hord

# python hord.py --disease fanconi --mlmodel morf --opt hyperopt --seed 42 --mode train  

# python rf.py
# python relevance_v2.py model
# python relevance_v2.py global
# python relevance_v2.py task
