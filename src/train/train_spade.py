import argparse
import ast

from trainers.spade_trainer import SPADETrainer

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r'/projects/0/prjs1204/data/mamamia/mamamia_processed/mamamia_cropped_96_train/images/', help="data directory.")
    parser.add_argument("--data_dir_val", type=str, default=r'/projects/0/prjs1204/data/mamamia/mamamia_processed/mamamia_cropped_96_valid/images/', help="data directory.")
    parser.add_argument("--df_train_valid", type=str, default=r'/projects/0/prjs1204/projects/spadebreast2025/src/spadebreast/dataset/combined_df/stratified_96_train_valid_df_seg.csv', help="data directory.")
    parser.add_argument("--df_train", type=str, default=r'/projects/0/prjs1204/projects/spadebreast2025/src/spadebreast/dataset/combined_df/stratified_96_train_df_seg.csv', help="data directory.")
    parser.add_argument("--df_valid", type=str, default=r'/projects/0/prjs1204/projects/spadebreast2025/src/spadebreast/dataset/combined_df/stratified_96_valid_df_seg.csv', help="data directory.")

    
    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir",default='./checkpoints', help="Location for models.")
    
    
    parser.add_argument(
        "--spatial_dimension", default=3, type=int, help="Dimension of images: 2d or 3d."
    )
    parser.add_argument("--image_size", default=None, help="Resize images.")
    parser.add_argument(
        "--image_roi",
        default=None,
        help="Specify central ROI crop of inputs, as a tuple, with -1 to not crop a dimension.",
        type=ast.literal_eval,
    )

    # model params
    parser.add_argument("--spade_in_channels", default=1, type=int)
    parser.add_argument("--spade_out_channels", default=1, type=int)
    # parser.add_argument("--spade_num_channels", default=(64, 64, 128, 256), type=ast.literal_eval)
    parser.add_argument("--spade_num_channels", default=(64, 128, 256), type=ast.literal_eval)

    parser.add_argument("--G_learning_rate", default=5e-5, type=float)
    parser.add_argument("--D_learning_rate", default=1e-4, type=float)
    # parser.add_argument("--spade_attention_levels", default=[False, False, False, True], type=ast.literal_eval)
    parser.add_argument("--spade_attention_levels", default=[False, False, False], type=ast.literal_eval)
    parser.add_argument("--spade_latent_channels", default=8, type=int)
    parser.add_argument("--spade_norm_num_groups", default=8, type=int)
    parser.add_argument("--spade_label_nc", default=2, type=int)
    parser.add_argument("--spade_num_res_blocks", default=3, type=int)


    
    # training param
    

    
    parser.add_argument("--n_epochs", type=int, default=210, help="Number of epochs to train.")
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=5,
        help="Number of epochs to between evaluations.",
    )
    parser.add_argument(
        "--augmentation",
        type=int,
        default=1,
        help="Use of augmentation, 1 (True) or 0 (False).",
    )
    parser.add_argument(
        "--adversarial_weight",
        type=float,
        default=0.01,
        help="Weight for adversarial component.",
    )
    parser.add_argument(
        "--adversarial_warmup",
        type=int,
        default=20,
        help="Warmup numbe ot epochs for the adversarial component.",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="Number of loader workers, set to 0 for windows.")
    parser.add_argument(
        "--cache_data",
        type=int,
        default=1,
        help="Whether or not to cache data in dataloaders.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=20,
        help="Save a checkpoint every checkpoint_every epochs.",
    )
    parser.add_argument(
        "--quick_test",
        default=0,
        type=int,
        help="If True, runs through a single batch of the train and eval loop.",
    )
    parser.add_argument("--project", type=str, default='breastMRI_spade2025', help="project name")
    parser.add_argument("--resume", default=False, type=bool , help="resume training from the last checkpoint")
    parser.add_argument("--resume_epoch", default=120, help="resume training from the specific epoch")
    parser.add_argument("--gpu_idx", type=str, default="2", help="Comma-separated list of GPU indices to use (e.g., '0,1,2').")
    parser.add_argument("--model_name", default='ddp_spade_96_250409', help="Name of model.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--device_index", type=int, default=2, help="gpu index.")
    parser.add_argument("--wandb", default=False, type=bool , help="if use wandb")

    parser.add_argument("--train_valid_combo", default=True, type=bool , help="if use wandb")    

    args = parser.parse_args()
    return args
# --node_rank=0
# to run using DDP, run set CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2  train_vqvae.py --model_name "AutoKL_240812_ddp"
# $env:CUDA_VISIBLE_DEVICES="2,3"
# torchrun --standalone --nproc_per_node=2 train_vqvae.py --model_name "AutoKL_240812_ddp"
if __name__ == "__main__":
    
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    trainer = SPADETrainer(args)
    trainer.train(args)

