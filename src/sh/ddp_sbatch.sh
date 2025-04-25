#!/bin/bash  
 
#SBATCH --partition=gpu_a100            #gpu_a100  gpu #SBATCH --mem=30G
#SBATCH --nodes=1                         # Number of nodes (1 node in this case)
#SBATCH --ntasks=4                        # Number of tasks (one per GPU, 2 tasks in total)
#SBATCH --gpus-per-node=1                # Number of GPUs per node
#SBATCH --cpus-per-task=18                # CPU cores per GPU (adjust based on your system)
#SBATCH --mem-per-gpu=38G
#SBATCH --time=2-10:05:00
#SBATCH --job-name=spade_2025_ddp
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

# torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 train_spade.py --batch_size 2 --model_name 'ddp_spade_96_250409 

# --train_valid_combo combining trainig data with validation data
# _2 uses two donssampling for increasing the quality of the reconstruction
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 train_spade.py --batch_size 2 --model_name 'ddp_spade_96_250414_2' 
