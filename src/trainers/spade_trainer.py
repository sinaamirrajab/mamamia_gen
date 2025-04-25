import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
# from monai.losses.spectral_loss import JukeboxLoss
from monai.networks.nets import PatchDiscriminator, AutoencoderKL
from monai.networks.nets import SPADEAutoencoderKL
from monai.networks.layers import Act
from torch.nn import L1Loss
import matplotlib
# from torch.optim.lr_scheduler import ReduceLROnPlateau
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

# from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
# from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from dataset.breastmri import BREASTSpadeDataset
from dataset.get_data_df import BREASTSpadeDatasetDF
from torch.utils.data import DataLoader
# from monai.config import print_config
from torch.cuda.amp import autocast
import warnings
# torch.autograd.set_detect_anomaly(True)
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# print_config()
import wandb
# https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/2d_spade_ldm/2d_spade_ldm.ipynb

# https://github.com/Project-MONAI/tutorials/blob/main/generation/maisi/maisi_train_vae_tutorial.ipynb
class SPADETrainer:
    def __init__(self, args):
        # initialise DDP if run was launched with torchrun
        self.wandb = args.wandb
        if "LOCAL_RANK" in os.environ:
            print("Setting up DDP.")
            # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
            self.ddp = True
            # disable logging for processes except 0 on every node
            local_rank = int(os.environ["LOCAL_RANK"])
            # if local_rank != 0:
            #     f = open(os.devnull, "w")
            #     sys.stdout = sys.stderr = f

            # # initialize the distributed training process, every GPU runs in a process nccl
            
            dist.init_process_group(backend="gloo")
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            print(f"Running on device {self.device}")
        else:
            self.ddp = False
            # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            # Check if CUDA is available and get the device index
            if torch.cuda.is_available():
                # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
                device_index = torch.cuda.current_device()
                # device_index = args.device_index
                self.device = torch.device(f"cuda:{device_index}")
                print(f"CUDA is available. Using device index: {device_index}")
                print(f"Device name: {torch.cuda.get_device_name(device_index)}")
            else:
                raise ValueError("CUDA is not available. Please check your installation.")
                self.device = torch.device("cpu")
                print("CUDA is not available. Using CPU.")
            
        torch.cuda.set_device(self.device)
        print(f"Running on device {self.device}")

        # print(f"Arguments: {str(args)}")
        # for k, v in vars(args).items():
        #     print(f"  {k}: {v}")

        # set up model
        self.spatial_dimension = args.spatial_dimension

        self.model = SPADEAutoencoderKL(
                                spatial_dims=args.spatial_dimension,
                                in_channels=args.spade_in_channels,
                                out_channels=args.spade_out_channels,
                                num_res_blocks =args.spade_num_res_blocks,
                                channels =args.spade_num_channels,
                                norm_num_groups=args.spade_norm_num_groups,
                                attention_levels=args.spade_attention_levels,
                                label_nc = args.spade_label_nc,
                            )
        self.model.to(self.device)
        print(f"{sum(p.numel() for p in self.model.parameters()):,} model parameters")

        self.discriminator = PatchDiscriminator(
            spatial_dims=args.spatial_dimension,
            num_layers_d=3,
            channels=64,
            in_channels=args.spade_in_channels,
            out_channels=args.spade_out_channels,
            kernel_size=4,
            activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
            norm="INSTANCE",
            bias=False,
            padding=1,
        )
        self.discriminator.to(self.device)
        print(f"{sum(p.numel() for p in self.discriminator.parameters()):,} discriminator parameters")
        # medicalnet_resnet50_23datasets
        self.perceptual_loss = PerceptualLoss(
            spatial_dims=args.spatial_dimension,  network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
        self.perceptual_loss.to(self.device)
        # self.jukebox_loss = JukeboxLoss(spatial_dims=args.spatial_dimension)
        # self.jukebox_loss.to(self.device)
        self.optimizer_g = torch.optim.Adam(
            params=self.model.parameters(), lr=args.G_learning_rate
        )
        self.optimizer_d = torch.optim.Adam(params=self.discriminator.parameters(), lr=args.D_learning_rate)
        # self.scheduler = ReduceLROnPlateau(self.optimizer_g, mode='min', factor=0.1, patience=5, verbose=True)
        self.G_learning_rate = args.G_learning_rate
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer_g,  lr_lambda=self.linear_lr_lambda)


        self.l1_loss = L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.adv_weight = args.adversarial_weight
        self.perceptual_weight = 0.3  #it was 0.001
        self.kl_weight = 1e-6
        self.autoencoder_warm_up_n_epochs = args.adversarial_warmup

        self.scaler_g = torch.cuda.amp.GradScaler(init_scale=2.0**8, growth_factor=1.5)
        self.scaler_d = torch.cuda.amp.GradScaler(init_scale=2.0**8, growth_factor=1.5)
        # set up optimizer, loss, checkpoints
        os.makedirs(args.output_dir, exist_ok=True)
        self.run_dir = Path(args.output_dir) / args.model_name
        os.makedirs(self.run_dir, exist_ok=True)
        if args.resume:
            print(f"Resuming training from checkpoint at epoch {args.resume_epoch}.")
            checkpoint_path = self.run_dir / f"checkpoint_{args.resume_epoch}.pth"
        else:
            checkpoint_path = self.run_dir / "checkpoint.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.start_epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint["global_step"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
            self.optimizer_g.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_loss = checkpoint["best_loss"]
            print(
                f"Resuming training using checkpoint {checkpoint_path} at epoch {self.start_epoch}"
            )
        else:
            self.start_epoch = 0
            self.best_loss = 1000
            self.global_step = 0

        # save vqvae parameters
        self.run_dir.mkdir(exist_ok=True)
       
        with open(self.run_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
       
        # wrap the model with DistributedDataParallel module
        if self.ddp:
            print("Wrapping model with DistributedDataParallel.")
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                find_unused_parameters=True

            )
            self.discriminator = DistributedDataParallel(
                self.discriminator,
                device_ids=[self.device],
                find_unused_parameters=True
            )

        if args.quick_test:
            print("Quick test enabled, only running on a single train and eval batch.")
        self.quick_test = args.quick_test
        # self.logger_train = SummaryWriter(log_dir=str(self.run_dir / "train"))
        # self.logger_val = SummaryWriter(log_dir=str(self.run_dir / "val"))
        self.num_epochs = args.n_epochs
        if args.train_valid_combo:
            train_dataset = BREASTSpadeDatasetDF(df=args.df_train_valid, phase="train")
        else:
            train_dataset = BREASTSpadeDatasetDF(df=args.df_train, phase="train")
        valid_dataset = BREASTSpadeDatasetDF(df=args.df_valid, phase="valid")
        if self.ddp:
            self.train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.batch_size,
                                  num_workers=args.num_workers, sampler=DistributedSampler(train_dataset))
            self.val_loader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=args.batch_size,
                                  num_workers=args.num_workers, sampler=DistributedSampler(valid_dataset, shuffle=False))
        else:
            self.train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
            self.val_loader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        print("size of train dataset: ", len(train_dataset))
        
        
        print("size of valid dataset: ", len(valid_dataset))
        if self.wandb:

           wandb.init(project=args.project, group=args.model_name, config=args, entity="sinaamirrajab")
    # Define a Lambda function for the scheduler
    def linear_lr_lambda(self, epoch):
        decay_start_epoch = 150
        linear_decay_epochs = 50
        lr_start = self.G_learning_rate
        lr_end = 1e-7
        if epoch < decay_start_epoch:
            return 1.0  # No change in learning rate for the first (num_epochs - 50) epochs
        else:
            # Linear decay factor from 1.0 to the ratio of lr_end/lr_start
            return 1 - (epoch - decay_start_epoch) / linear_decay_epochs * (1 - lr_end / lr_start)

    def one_hot(self, input_label, label_nc):
        # One hot encoding function for the labels
        shape_ = list(input_label.shape)
        shape_[1] = label_nc
        label_out = torch.zeros(shape_)
        for channel in range(label_nc):
            label_out[:, channel, ...] = input_label[:, 0, ...] == channel
        return label_out

    def save_checkpoint(self, path, epoch, save_message=None):
        if self.ddp and dist.get_rank() == 0:
            # if DDP save a state dict that can be loaded by non-parallel models
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.module.state_dict(),
                "discriminator_state_dict": self.discriminator.module.state_dict(),
                "optimizer_state_dict": self.optimizer_g.state_dict(),
                "optimizer_d_state_dict": self.optimizer_d.state_dict(),
                "best_loss": self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)
        if not self.ddp:
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "optimizer_state_dict": self.optimizer_g.state_dict(),
                "optimizer_d_state_dict": self.optimizer_d.state_dict(),
                "best_loss": self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)

    def train(self, args):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            epoch_loss = self.train_epoch(epoch)
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

                self.save_checkpoint(
                    self.run_dir / "checkpoint.pth",
                    epoch,
                    save_message=f"Saving checkpoint for model with loss {self.best_loss}",
                )

            if args.checkpoint_every != 0 and (epoch + 1) % args.checkpoint_every == 0:
                self.save_checkpoint(
                    self.run_dir / f"checkpoint_{epoch+1}.pth",
                    epoch,
                    save_message=f"Saving checkpoint at epoch {epoch+1}",
                )

            if (epoch + 1) % args.eval_freq == 0:
                self.model.eval()
                self.val_epoch(epoch)
        print("Training completed.")
        if self.wandb:
            wandb.finish()
        if self.ddp:
            dist.destroy_process_group()

    def train_epoch(self, epoch):
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            ncols=150,
            position=0,
            leave=True,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        l1_loss = 0
        generator_epoch_loss = 0
        discriminator_epoch_loss = 0
        epoch_step = 0
        self.model.train()
        for step, batch in progress_bar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            labels_one_hot = self.one_hot(labels, 2).to(self.device)
            # self.check_for_nan_and_raise(labels_one_hot)
            # self.check_for_nan_and_raise(images)
            self.optimizer_g.zero_grad(set_to_none=True)

            # Generator part
            with autocast(enabled=True):
            # reconstruction, quantization_loss = self.model(images=images)
                reconstruction, z_mu, z_sigma = self.model(images, labels_one_hot)
                # self.check_for_nan_and_raise(reconstruction)
                

                recons_loss = self.l1_loss(reconstruction.float(), images.float())

                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                perceptual_loss = self.perceptual_loss(reconstruction.float(), images.float())
                # jukebox_loss = self.jukebox_loss(reconstruction.float(), images.float())
                total_generator_loss = (
                recons_loss
                # + quantization_loss
                + self.kl_weight * kl_loss
                + self.perceptual_weight * perceptual_loss
                # + jukebox_loss
                # + adv_weight * adversarial_loss
                    )
                if epoch > self.autoencoder_warm_up_n_epochs:
                    logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
                    adversarial_loss = self.adv_loss(
                    logits_fake, target_is_real=True, for_discriminator=False
                    )
                    total_generator_loss += self.adv_weight * adversarial_loss


            self.scaler_g.scale(total_generator_loss).backward()
            # Gradient clipping
            clip_value=1.0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()

            # Discriminator part
            if epoch > self.autoencoder_warm_up_n_epochs:
                self.optimizer_d.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = self.discriminator(images.contiguous().detach())[-1]
                    loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                    loss_d = self.adv_weight * discriminator_loss

                self.scaler_d.scale(loss_d).backward()
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()

            l1_loss += recons_loss.item()
            generator_epoch_loss += total_generator_loss.item()
            if epoch > self.autoencoder_warm_up_n_epochs:
                discriminator_epoch_loss += discriminator_loss.item()
            epoch_step += images.shape[0]
            self.global_step += images.shape[0]
            progress_bar.set_postfix(
                {
                    "l1_loss": l1_loss / (epoch_step),
                    "generator_loss": generator_epoch_loss / (epoch_step),
                    "discriminator_loss": discriminator_epoch_loss / (epoch_step),
                }
            )
            if self.wandb:
                wandb.log({
                        "recons_loss": l1_loss / (epoch_step),
                        "total_generator_loss": total_generator_loss.item(),
                        "gen_adv_loss": adversarial_loss.item() if epoch > self.autoencoder_warm_up_n_epochs else 0,
                        "disc_loss": discriminator_loss.item() if epoch > self.autoencoder_warm_up_n_epochs else 0,
                        "pereceptual_loss": perceptual_loss.item(),
                    })

        
            if step%500== 0: # log to wandb
                fig, axes = plt.subplots(1, 4, figsize=(8, 2))
                i_n = 0
                # for i in range(i_n):
                for j in range(4):
                    slice_idx = int(images.shape[-1]/2 + j*10)
                    image_to_show = images[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                    label_to_show = labels[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                    rec_to_show = reconstruction[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                    axes[j].imshow(np.concatenate([image_to_show, label_to_show, rec_to_show],axis=1), cmap='gray')      
                    axes[j].axis('off')
                # axes[j].set_title(batch['data']['image_id'][i_n])
                # plt.subplots_adjust(wspace=0.1, hspace=0.1)
                plt.tight_layout()
                os.makedirs(os.path.join(self.run_dir, 'train_img'), exist_ok=True)
                file_path = os.path.join(self.run_dir, 'train_img', f"epoch_{epoch}_step_{step}.png")
                plt.savefig(file_path)
                plt.close(fig)
                # if self.wandb:
                #     wandb.log({"training_image_recon": fig})
           

            if self.quick_test:
                break
        epoch_loss = l1_loss / epoch_step
        
        self.scheduler.step()
        print(f'learning rate is {self.scheduler.get_last_lr()[0]}')
        
        return epoch_loss
    def check_for_nan_and_raise(self, data):
        """Checks for NaN values in a PyTorch tensor and raises an error if found.

        Args:
            data: The PyTorch tensor to check.
        """

        if torch.isnan(data).any():
            raise ValueError("NaN values found in data")
    def val_epoch(self, epoch):
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                ncols=150,
                position=0,
                leave=True,
                desc="Validation",
            )

            global_val_step = self.global_step
            val_loss = 0
            epoch_step = 0
            for step, batch in progress_bar:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                labels_one_hot = self.one_hot(labels, 2).to(self.device)
                # reconstruction, quantization_loss = self.model(images=images)
                reconstruction, z_mu, z_sigma = self.model(images, labels_one_hot)
                logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]

                recons_loss = self.l1_loss(reconstruction.float(), images.float())
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                perceptual_loss = self.perceptual_loss(reconstruction.float(), images.float())
                # jukebox_loss = self.jukebox_loss(reconstruction.float(), images.float())
                adversarial_loss = self.adv_loss(
                    logits_fake, target_is_real=True, for_discriminator=False
                )
                total_generator_loss = (
                    recons_loss
                    # + quantization_loss
                    + self.kl_weight * kl_loss
                    + self.perceptual_weight * perceptual_loss
                    # + jukebox_loss
                    + self.adv_weight * adversarial_loss
                )
                

                global_val_step += images.shape[0]
                val_loss += recons_loss.item()
                epoch_step += images.shape[0]
                # plot some recons
                if step== 0: # log to wandb
                    fig, axes = plt.subplots(1, 4, figsize=(8, 2))
                    i_n = 0
                    # for i in range(i_n):
                    for j in range(4):
                        slice_idx = int(images.shape[-1]/2 + j*10)
                        image_to_show = images[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                        label_to_show = labels[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                        rec_to_show = reconstruction[i_n, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                        axes[j].imshow(np.concatenate([image_to_show, label_to_show, rec_to_show],axis=1), cmap='gray')      
                        axes[j].axis('off')
                    # axes[j].set_title(batch['data']['image_id'][i_n])
                    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
                    plt.tight_layout()
                    os.makedirs(os.path.join(self.run_dir, 'valid_img'), exist_ok=True)
                    file_path = os.path.join(self.run_dir, 'valid_img', f"epoch_{epoch}_step_{step}.png")
                    plt.savefig(file_path)
                    plt.close(fig)
                    
                    # wandb.log({"validation_image_recon": fig})
                if self.wandb:
                    wandb.log({
                            "validation_recons_loss": val_loss/epoch_step,
                        })
                if self.quick_test:
                    break
                # val_loss /= epoch_step
                
        