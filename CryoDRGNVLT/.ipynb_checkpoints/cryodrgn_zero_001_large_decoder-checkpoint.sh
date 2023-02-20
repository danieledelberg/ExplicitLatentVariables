#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --partition=pi_lederman
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --output="%j.out"

module load miniconda
conda deactivate
conda activate cryodrgn_rotations

cryodrgn train_vae particles.128.mrcs \
                   --ctf ctf.pkl \
                   --poses poses.pkl \
                   --zdim 8 \
                   -n 50 \
                   --uninvert-data \
                   --zlr 0.001 \
                   --zinit zero \
                   --dec-layers 5 \
                   --dec-dim 256 \
                   -o data/zero_001_large_decoder/ > data/zero_001_large_decoder.log