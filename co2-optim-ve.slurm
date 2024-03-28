#!/bin/bash

#SBATCH -J CO2optim      # Job name
#SBATCH -o out.%j        # Name of stdout output file
#SBATCH -e err.%j        # Name of stderr error file
#SBATCH -p normal        # Queue (partition) name
#SBATCH -N 1             # Total # of nodes (must be 1 for serial)
#SBATCH -n 1             # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00      # Run time (hh:mm:ss)
#SBATCH -A EAR23030      # Project/Allocation name
#SBATCH --mail-type=all  # Send email at begin and end of job
#SBATCH --mail-user=misaelmorales@utexas.edu

# Startup environment
pwd
date
module load matlab

# Run the program
matlab hpcRunner