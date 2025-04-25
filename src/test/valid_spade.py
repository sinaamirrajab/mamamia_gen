# %%
import torch
import torch.nn.functional as F
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
# %%
# test dataloader
from dataset.breastmri import BREASTDataset, BREASTSpadeDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from monai.utils import set_determinism
set_determinism(seed=0)
def one_hot(input_label, label_nc):
  # One hot encoding function for the labels
  shape_ = list(input_label.shape)
  shape_[1] = label_nc
  label_out = torch.zeros(shape_)
  for channel in range(label_nc):
      label_out[:, channel, ...] = input_label[:, 0, ...] == channel
  return label_out

data_dir = r"F:\Sina\data\mamamia_cropped_96_valid\images"
num_workers  = 0
batch_size = 1
train_dataset = BREASTSpadeDataset(
            root_dir=data_dir)
train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size,
                                  num_workers=num_workers)
print("size of train dataset: ", len(train_dataset))

# %%
def get_image_from_dataloader(dataloader, index):
    # Create an iterator from the dataloader
    data_iter = iter(dataloader)
    
    # Calculate the batch index and the image index within the batch
    batch_size = dataloader.batch_size
    batch_index = index // batch_size
    image_index = index % batch_size
    
    # Iterate through the dataloader to get the desired batch
    for _ in range(batch_index + 1):
        batch = next(data_iter)
    
    image_name = batch['name'][image_index]
    
    return batch, image_name

batch, image_name = get_image_from_dataloader(train_dataloader, 3)
print(image_name)

# %%
################################################

labels = batch["label"]
print(labels.shape)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(labels[0,:, :,:, 20+i*5].squeeze().cpu().numpy(), cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()

eroded_mask = erode_mask(labels, kernel_size=2, iterations=1)
dilated_mask = dilate_mask(labels, kernel_size=3, iterations=1)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(eroded_mask[0,:, :,:, 20+i*5].squeeze().cpu().numpy(), cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(dilated_mask[0,:, :,:, 20+i*5].squeeze().cpu().numpy(), cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()


#%%
from train_spade import parse_args
from monai.networks.nets import SPADEAutoencoderKL

# the following line is necessary to avoid an error when parsing the arguments in the notebook > parser.parse_args()
import sys
sys.argv=['']
del sys
args = parse_args()

model = SPADEAutoencoderKL(
                                spatial_dims=args.spatial_dimension,
                                in_channels=args.spade_in_channels,
                                out_channels=args.spade_out_channels,
                                num_res_blocks =args.spade_num_res_blocks,
                                num_channels =args.spade_num_channels,
                                norm_num_groups=args.spade_norm_num_groups,
                                attention_levels=args.spade_attention_levels,
                                latent_channels=args.spade_latent_channels,
                                label_nc = args.spade_label_nc,
                            )

print(f"{sum(p.numel() for p in model.parameters()):,} model parameters")


# %%

import os
from pathlib import Path

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model.to(device)



run_dir = Path(args.output_dir) / args.model_name 


checkpoint_path = run_dir / "checkpoint.pth"
            


checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
vae_epoch = checkpoint["epoch"]

print(
                    f"VAE checkpoint {checkpoint_path} at epoch {vae_epoch}"
                )






# %%
# %matplotlib inline
from torch.cuda.amp import autocast
from tqdm import tqdm

model.eval()
# text_encoder

with autocast(enabled=True):
    with torch.no_grad():
        # batch = next(iter(train_dataloader))
        
        images = batch["image"].to(device)
                    
        labels = batch["label"].to(device)
        labels_one_hot = one_hot(labels, 2).to(device)
        dilated_mask_oh = one_hot(dilated_mask, 2).to(device)
        eroded_mask_oh = one_hot(eroded_mask, 2).to(device)

        z = model.encode_stage_2_inputs(images)
        recon = model.decode_stage_2_outputs(z, seg=labels_one_hot)
        dilated_recon = model.decode_stage_2_outputs(z, seg=dilated_mask_oh)
        eroded_recon = model.decode_stage_2_outputs(z, seg=eroded_mask_oh)
        

        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))

        for i, ax in enumerate(axes.flatten()):
            ax.imshow(images[0,:, :,:, 20+i*5].squeeze().cpu().numpy(), cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))

        for i, ax in enumerate(axes.flatten()):
            ax.imshow(recon[0,:, :,:, 20+i*5].squeeze().cpu().numpy(), cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))

        for i, ax in enumerate(axes.flatten()):
            ax.imshow(dilated_recon[0,:, :,:, 20+i*5].squeeze().cpu().numpy(), cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))

        for i, ax in enumerate(axes.flatten()):
            ax.imshow(eroded_recon[0,:, :,:, 20+i*5].squeeze().cpu().numpy(), cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

# %%
print(batch['name'])
# %%

print(image_name)
# %%
# batch, image_name = get_image_from_dataloader(train_dataloader, 3)
# print(image_name)



plt.imshow(images[0,:, :,:, 50].squeeze().cpu().numpy(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
plt.imshow(labels[0,:, :,:, 50].squeeze().cpu().numpy(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.imshow(recon[0,:, :,:, 50].squeeze().cpu().numpy(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
# %%

plt.imshow(dilated_mask[0,:, :,:, 50].squeeze().cpu().numpy(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
plt.imshow(dilated_recon[0,:, :,:, 50].squeeze().cpu().numpy(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.imshow(eroded_mask[0,:, :,:, 50].squeeze().cpu().numpy(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
plt.imshow(eroded_recon[0,:, :,:, 50].squeeze().cpu().numpy(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
# %%
