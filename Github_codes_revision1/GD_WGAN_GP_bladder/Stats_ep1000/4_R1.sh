#!/bin/bash

#SBATCH --partition=general-gpu               # Name of Partition
#SBATCH --nodes=1                             # Ensure all cores are from the same node
#SBATCH --constraint=epyc64
#SBATCH --constraint=a100
#SBATCH --mem=256G                            # Request 256GB of available RAM
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate tf

python 4_My_similarity_R1.py
