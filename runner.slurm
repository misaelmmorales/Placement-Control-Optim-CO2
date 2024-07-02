#!/bin/bash

#SBATCH -J CO2Optim      # Job name
#SBATCH -o %j.out        # Name of stdout output file
#SBATCH -e %j.err        # Name of stderr error file
#SBATCH -p gpu-h100      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes (must be 1 for serial)
#SBATCH -n 1             # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00      # Run time (hh:mm:ss)
#SBATCH -A EAR23030      # Project/Allocation name
#SBATCH --mem=256G       # memory

# Startup environment
pwd
date

# Run the program
python3 main.py