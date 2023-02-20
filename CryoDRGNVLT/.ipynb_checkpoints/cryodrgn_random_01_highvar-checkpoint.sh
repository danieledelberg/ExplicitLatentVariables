#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --partition=scavenge_gpu
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
                   --uninvert-data \
                   --randomvar 0.0 \
                   --zlr 0.01 \
                   --zinit random \
                   -o datav2/random_01_highvar/ > datav2/random_01_highvar.log