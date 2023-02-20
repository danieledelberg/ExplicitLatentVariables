#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --output="%j.out"

module load miniconda
conda deactivate
conda activate cryodrgn_rotations

cd cryodrgn
python setup.py install
cd ../
cryodrgn train_vae particles.128.mrcs --ctf ctf.pkl --poses poses.pkl --zdim 8 -n 50 --uninvert-data -o data/pretrained_0001/ > data/pretrained_0001.log