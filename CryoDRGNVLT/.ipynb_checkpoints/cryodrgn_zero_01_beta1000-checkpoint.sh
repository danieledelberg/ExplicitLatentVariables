#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --partition=pi_lederman
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --output="%j.out"

module load miniconda
conda activate cryodrgn_rotations

cryodrgn train_vae particles.128.mrcs \
                   --ctf ctf.pkl \
                   --poses poses.pkl \
                   --zdim 8 \
                   -n 100 \
                   --seed 0 \
                   --beta 1000 \
                   --uninvert-data \
                   --zlr 0.01 \
                   --zinit zero \
                   --load datav2/zero_01_b1000/weights.43.pkl \
                   --loadz datav2/zero_01_b1000/z_vals.43.pkl \
                   -o datav2/zero_01_b1000/ > datav2/zero_01_b1000.log