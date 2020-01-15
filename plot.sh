BASE="/mnt/lustre/scratch/CBRA/research/projects/holrd/experiments/"

declare -a DISEASES
DISEASES=("ALB" "EVC" "H" "RP" "WS")
declare -a EXPERIMENTS
EXPERIMENTS=("AT" "AT_C")

## now loop through the above array
for DISEASE in "${DISEASES[@]}"
do
    for EXPERIMENT in "${EXPERIMENTS[@]}"
    do
        JOBNAME="plot_${DISEASE}_${EXPERIMENT}"
        MLPATH=$BASE/${DISEASE}/${EXPERIMENT}/ml/morf_train
        sbatch -J ${JOBNAME} --export=MLPATH=$MLPATH plot.sbatch
    done
   # or do whatever with individual element of the array
done
