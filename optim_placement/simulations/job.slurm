#!/bin/bash
#SBATCH -J co2leak
#SBATCH -N 10
#SBATCH -o co2out
#SBATCH -t 1:00:00

module load matlab/R2023a

matlab -nodisplay -nosplash -r MAIN3D