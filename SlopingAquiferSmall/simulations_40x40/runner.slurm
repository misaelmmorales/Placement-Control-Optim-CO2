#!/bin/bash

#SBATCH -J CO2sim         # Job name
#SBATCH -o logs/%A_%a.out # Name of stdout output file
#SBATCH -e logs/%A_%a.err # Name of stderr error file
#SBATCH -p normal         # Queue (partition) name
#SBATCH -N 1              # Total # of nodes
#SBATCH -n 1              # Total # of mpi tasks
#SBATCH -c 1              # Number of CPUs per task
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH -A EAR23030       # Project/Allocation name
#SBATCH --mem=64G         # Memory per node
#SBATCH --array=1-24      # Array range (24 jobs to cover 1272 simulations)

module purge
module load matlab

export OMP_NUM_THREADS=1

# Startup environment
pwd
date
echo "Running job with SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"

# Calculate the base start index for this job
BASE_INDEX=$(( (${SLURM_ARRAY_TASK_ID} - 1) * 53 + 1 ))

# Loop over 53 simulations per job
for (( i=${BASE_INDEX}; i<${BASE_INDEX}+53; i++ )); do
  srun --exclusive -n 1 -c 1 matlab -nodisplay -r "hpc_runSlopingAquiferBig(${i}); exit;" &
done

wait  # Wait for all MATLAB instances to complete
