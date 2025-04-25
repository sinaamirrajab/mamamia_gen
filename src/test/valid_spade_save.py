



from monai.networks.nets import SPADEAutoencoderKL
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from dataset.breastmri import BREASTSpadeDataset
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTextModel
import os
from pathlib import Path
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from monai.networks.schedulers import DDPMScheduler
from tqdm import tqdm
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

def inference(args, phase: str = "valid"):

    ## load data
    if phase == "valid":
        train_dataset = BREASTSpadeDataset(
                root_dir=args.data_dir_val)
        train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.batch_size,
                                        num_workers=args.num_workers)
        print("size of train dataset: ", len(train_dataset))
    elif phase == "train":
        train_dataset = BREASTSpadeDataset(
                root_dir=args.data_dir)
        train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.batch_size,
                                        num_workers=args.num_workers)
        print("size of train dataset: ", len(train_dataset))
    else:
        raise ValueError("Invalid phase. Must be 'train' or 'valid'.")

    # model architecture
    model = SPADEAutoencoderKL(
                                spatial_dims=args.spatial_dimension,
                                in_channels=args.spade_in_channels,
                                out_channels=args.spade_out_channels,
                                num_res_blocks =args.spade_num_res_blocks,
                                num_channels =args.spade_num_channels,
                                norm_num_groups=args.spade_norm_num_groups,
                                attention_levels=args.spade_attention_levels,
                                label_nc = args.spade_label_nc,
                            )

    print(f"{sum(p.numel() for p in model.parameters()):,} model parameters")


    # load trained models
    if torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        device_index = torch.cuda.current_device()
        # device_index = args.device_index
        device = torch.device(f"cuda:{device_index}")
        print(f"CUDA is available. Using device index: {device_index}")
        print(f"Device name: {torch.cuda.get_device_name(device_index)}")
    else:
        raise ValueError("CUDA is not available. Please check your installation.")


    model.to(device)



  
    run_dir = Path(args.output_dir) / args.model_name 


    checkpoint_path = run_dir / "checkpoint.pth"
                
            


    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    vae_epoch = checkpoint["epoch"]

    print(
                        f"VAE checkpoint {checkpoint_path} at epoch {vae_epoch}"
                    )



    # model inference


    model.eval()

    with autocast(enabled=True):
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                ncols=150,
                position=0,
                leave=True,
                desc="Validation",
            )
            for step, batch in progress_bar:
                images = batch["image"].to(device)
            
            
    
                        
                labels = batch["label"].to(device)
                labels_one_hot = one_hot(labels, 2).to(device)

                z = model.encode_stage_2_inputs(images)
                # scale_factor = 1
                scale_factor = 1 / torch.std(z)
                z = z * scale_factor

                    # plt.show()
                out_name = batch['name'][0].split('.')[0]
                os.makedirs(os.path.join(run_dir, 'encoded_images', phase), exist_ok=True)
                file_path = os.path.join(run_dir, 'encoded_images',phase,  f"{out_name}.pt")
                # Save the tensor z
                torch.save(z, file_path)
                    

                

if __name__ == "__main__":
    from train_spade_ldm import parse_args
    args = parse_args()
    args.batch_size = 1
    inference(args, phase = "valid")
    inference(args, phase = "train")
    print("Inference done!")