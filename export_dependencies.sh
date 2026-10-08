#!/bin/bash

#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH --output=export_dependencies_%j.out

module purge
source $HOME/miniconda3/etc/profile.d/conda.sh

echo "Exporting tf environment..."
conda activate tf
conda env export --no-builds > environment_tf.yml
python --version > python_tf_version.txt

echo "Exporting PepINVENT environment..."
conda activate reinvent4
conda env export --no-builds > environment_pepinvent.yml
python --version > python_pepinvent_version.txt

echo "Done."
