#!/bin/bash  
 
#SBATCH --partition=gpu_a100            #gpu_a100  gpu #SBATCH --mem=30G
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=38G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=3-01:05:00
#SBATCH --job-name=spade_2025
#SBATCH --output=log/%x_%j.dat
#SBATCH --error=log/%x_%j.dat
#SBATCH --open-mode=append

 
# load needed modules module purge
module load 2023
# module load Miniconda3/23.5.2-0
module load CUDA/12.1.1
 
# run the application using mpirun in this case

# Activate your Conda environment
source activate monai                  # Activate the bbox conda environment
# Set the environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Run the Python script

# python train_spade.py --batch_size 2 --model_name 'spade_96_250403'
python test_spade.py 
#  