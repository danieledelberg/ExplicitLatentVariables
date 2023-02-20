#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --partition=pi_lederman
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --output="%j-test.out"

module load miniconda
conda activate cryodrgn_rotations

# Set up the new changes
cd cryodrgn
python setup.py install
cd ..

# Run test
cryodrgn train_vae particles.test.mrcs \
                   --ctf ctf_test.pkl \
                   --poses poses_test.pkl \
                   --zdim 3 \
                   -n 3 \
                   --seed 0 \
                   --uninvert-data \
                   --zlr 0.01 \
                   --zinit random \
                   -o data/testing/ > data/testing.log