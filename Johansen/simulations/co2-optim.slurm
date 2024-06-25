#!/bin/bash

#SBATCH -J CO2optim      # Job name
#SBATCH -t 48:00:00      # Run time (hh:mm:ss)
#SBATCH -A EAR23030      # Project/Allocation name
#SBATCH -o %j.out        # Name of stdout output file
#SBATCH -e %j.err        # Name of stderr error file
#SBATCH -p gpu-h100      # Queue (partition) name
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=10
#SBATCH --array=0-9      
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=misaelmorales@utexas.edu

# Startup environment
module load matlab
pwd
date

N=1272
NUM_TASKS=10
NUM_CORES=10
REALIZATIONS_PER_TASK=$((N / (NUM_TASKS * SLURM_ARRAY_TASK_COUNT) ))
OFFSET=$((SLURM_ARRAY_TASK_ID * REALIZATIONS_PER_TASK * NUM_TASKS))

# Run the program
for task_id in $(seq 0 $((NUM_TASKS-1))); do
	TASK_OFFSET=$((OFFSET+task_id * REALIZATIONS_PER_TASK))
	srun --exclusive -N1 -n1 matlab -nodisplay -r "parRunner($TASK_OFFSET, $REALIZATIONS_PER_TASK); exit" &
done
wait
