# %%

import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.get_data_df import BREASTSpadeDatasetDF_valid
from torch.utils.data import DataLoader
import torchio as tio
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# %matplotlib inline
# Binary mask of size b x 1 x 128 x 128 x 64
# Assume `mask` is your binary mask tensor (dtype should be float or long)
# For example: mask = torch.randint(0, 2, (b, 1, 128, 128, 64)).float()

def erode_mask(mask, kernel_size=3, iterations=1):
    # Ensure the mask is of type float32
    mask = mask.float()

    # Create a 3D structuring element (kernel) for erosion, same type as mask
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=mask.device, dtype=mask.dtype)

    # Padding to maintain output size
    padding = (kernel_size - 1) // 2
    
    for _ in range(iterations):
        # Apply 3D convolution as erosion (min operation is simulated by convolution)
        mask = F.conv3d(mask, kernel, padding=padding)
        
        # Threshold the result back into binary (1 for the regions where all elements are 1, otherwise 0)
        mask = (mask == kernel.sum()).float()

    return mask

def dilate_mask(mask, kernel_size=3, iterations=1):
    # Ensure the mask is of type float32
    mask = mask.float()

    # Create a 3D structuring element (kernel) for dilation, same type as mask
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=mask.device, dtype=mask.dtype)

    # Padding to maintain output size
    padding = (kernel_size - 1) // 2
    
    for _ in range(iterations):
        # Apply 3D convolution as dilation (max operation simulated by convolution)
        mask = F.conv3d(mask, kernel, padding=padding)
        
        # Threshold the result back into binary (1 if any part of the kernel overlaps with 1, otherwise 0)
        mask = (mask > 0).float()

    return mask



def elastic_deformation_torchio(mask, num_control_points=7, max_displacement=7.5, locked_borders=2):
    """
    Apply elastic deformation to a 3D mask using TorchIO.

    Args:
        mask (torch.Tensor): 5D tensor of shape (B, 1, D, H, W)
        num_control_points (int): Number of control points for the B-spline grid.
        max_displacement (float): Max displacement at each control point (in voxels).
        locked_borders (int): Number of border control points to lock (avoid edge deformation).

    Returns:
        torch.Tensor: Elastically deformed mask tensor (same shape).
    """
    # Ensure float and move to CPU for torchio compatibility
    mask = mask.squeeze(1).float().cpu()

    # TorchIO Subject setup
    subject = tio.Subject(
        mask=tio.LabelMap(tensor=mask)  # use LabelMap for masks
    )

    # Define the transform
    transform = tio.RandomElasticDeformation(
        num_control_points=num_control_points,
        max_displacement=max_displacement,
        locked_borders=locked_borders,
        image_interpolation='nearest'  # keep mask discrete
    )

    # Apply the transform
    transformed = transform(subject)

    # Return the deformed mask as float tensor on original device
    return transformed.mask.tensor.to(mask.device).unsqueeze(1)


def one_hot(input_label, label_nc):
        # One hot encoding function for the labels
        shape_ = list(input_label.shape)
        shape_[1] = label_nc
        label_out = torch.zeros(shape_)
        for channel in range(label_nc):
            label_out[:, channel, ...] = input_label[:, 0, ...] == channel
        return label_out

#%%
from train_spade import parse_args
if __name__ == "__main__":
    
    

    from monai.networks.nets import SPADEAutoencoderKL

    # the following line is necessary to avoid an error when parsing the arguments in the notebook > parser.parse_args()
    import sys
    sys.argv=['']
    del sys
    args = parse_args()


    if torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        device_index = torch.cuda.current_device()
        # device_index = args.device_index
        device = torch.device(f"cuda:{device_index}")
        print(f"CUDA is available. Using device index: {device_index}")
        print(f"Device name: {torch.cuda.get_device_name(device_index)}")
    else:
        raise ValueError("CUDA is not available. Please check your installation.")
        self.device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
            
    torch.cuda.set_device(device)
    print(f"Running on device {device}")
    spatial_dimension = args.spatial_dimension

    model = SPADEAutoencoderKL(
                                spatial_dims=args.spatial_dimension,
                                in_channels=args.spade_in_channels,
                                out_channels=args.spade_out_channels,
                                num_res_blocks =args.spade_num_res_blocks,
                                channels =args.spade_num_channels,
                                norm_num_groups=args.spade_norm_num_groups,
                                attention_levels=args.spade_attention_levels,
                                label_nc = args.spade_label_nc,
                            )
    model.to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} model parameters")

    # Load the model
    
    run_dir = Path(args.output_dir) / args.model_name 


    checkpoint_path = run_dir / "checkpoint.pth"
                


    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    vae_epoch = checkpoint["epoch"]

    print(
                        f"VAE checkpoint {checkpoint_path} at epoch {vae_epoch}"
                    )
    
    
    train_dataset = BREASTSpadeDatasetDF_valid(df=args.df_train, phase="train")

    train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.batch_size,
                                num_workers=args.num_workers)
    
    print("size of train dataset: ", len(train_dataset))

    # Get a batch of data
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                ncols=150,
                position=0,
                leave=True,
            )
        for step, batch in progress_bar:
            if step > 100:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            di_labels = dilate_mask(labels, kernel_size=3, iterations=1)
            er_labels = erode_mask(labels, kernel_size=3, iterations=1)
            de_labels = elastic_deformation_torchio(labels, num_control_points=7, max_displacement=7.5, locked_borders=2)
            labels_oh = one_hot(labels, 2).to(device)
            di_labels_oh = one_hot(di_labels, 2).to(device)
            er_labels_oh = one_hot(er_labels, 2).to(device)
            de_labels_oh = one_hot(de_labels, 2).to(device)

            recon, z_mu, z_sigma = model(images, labels_oh)
            di_recon, _, _ = model(images, di_labels_oh)
            er_recon, _, _ = model(images, er_labels_oh)
            de_recon, _, _ = model(images, de_labels_oh)

            if step%10== 0: # log to wandb
                fig, axes = plt.subplots(1, 4, figsize=(8, 2))
                i_n = 0
                im_name = batch['data']['image_id'][i_n]
                # for i in range(i_n):
                for j in range(4):
                    slice_idx = int(images.shape[-1]/2 + j*10)
                    image_to_show = images[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                    label_to_show = labels[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                    rec_to_show = recon[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                    di_label_to_show = di_labels[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                    er_label_to_show = er_labels[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                    de_label_to_show = de_labels[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                    di_recon_to_show = di_recon[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                    er_recon_to_show = er_recon[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                    de_recon_to_show = de_recon[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                    or_con = np.concatenate([image_to_show, label_to_show, rec_to_show],axis=1)
                    di_con = np.concatenate([image_to_show, di_label_to_show, di_recon_to_show],axis=1)
                    er_con = np.concatenate([image_to_show, er_label_to_show, er_recon_to_show],axis=1)
                    de_con = np.concatenate([image_to_show, de_label_to_show, de_recon_to_show],axis=1)
                    axes[j].imshow(np.concatenate([or_con, di_con, er_con, de_con],axis=0), cmap='gray')   
                    axes[j].axis('off')
                axes[j].set_title(im_name)
                # plt.subplots_adjust(wspace=0.1, hspace=0.1)
                plt.tight_layout()
                os.makedirs(os.path.join(run_dir, "generation/example" ), exist_ok=True)
                file_path = os.path.join(run_dir, "generation/example", f"{im_name}_step_{step}.png")
                plt.savefig(file_path)
                plt.close(fig)
                    



