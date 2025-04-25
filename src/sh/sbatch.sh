#!/bin/bash  
 
#SBATCH --partition=gpu            #gpu_a100  gpu #SBATCH --mem=30G
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=38G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=3-01:05:00
#SBATCH --job-name=BreastVAE
#SBATCH --output=log/%x_%j.dat
#SBATCH --error=log/%x_%j.dat
#SBATCH --open-mode=append

 
# load needed modules module purge
module load 2023
# module load Miniconda3/23.5.2-0
module load CUDA/12.1.1
 
# run the application using mpirun in this case

# Activate your Conda environment
source activate spade                  # Activate the bbox conda environment
# Set the environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the Python script
# python test_gpus.py                   # Run the test script
# python train_spade.py
# python test_dataloader.py
################################### train autoencoder ###################################
python train_vqvae.py --batch_size 2

##################################### tarin and valid SPADE LDM #####################################
# python train_spade_ldm.py --batch_size 120
# python valid_spade_ldm.py --ldm_name 'spade_ldm_pt_20241009'
# python valid_spade_ldm.py
# python valid_spade_save.py
# python valid_spade_ldm.py --ldm_name 'spade_ldm_pt_00001_01_beta_lr001_20241016'
# python valid_spade_ldm.py --ldm_name 'spade_ldm_pt_v_00001_001_beta_lr001_20241016'
# python valid_spade_ldm.py --ldm_name 'spade_ldm_pt_vp_20241009'
# python valid_spade_ldm.py --ldm_name 'spade_ldm_pt_vp_20241010'

