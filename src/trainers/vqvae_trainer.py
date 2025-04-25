import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
# from generative.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses import PatchAdversarialLoss, PerceptualLoss, JukeboxLoss
# from generative.losses.perceptual import PerceptualLoss
# from generative.losses.spectral_loss import JukeboxLoss
# from generative.networks.nets import VQVAE, PatchDiscriminator, AutoencoderKL
from monai.networks.nets import PatchDiscriminator
from monai.networks.layers import Act
from torch.nn import L1Loss
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

# from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset.breastmri import BREASTDataset
from torch.utils.data import DataLoader
from monai.config import print_config
# print_config()
import wandb
# referene: https://github.com/evihuijben/ddpm_ood/blob/main/src/trainers/vqvae_trainer.py

class VQVAETrainer:
    def __init__(self, args):

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


        # set up model
        self.spatial_dimension = args.spatial_dimension
        vqvae_args = {
            "spatial_dims": args.spatial_dimension,
            "in_channels": args.vqvae_in_channels,
            "out_channels": args.vqvae_out_channels,
            "num_res_layers": args.vqvae_num_res_layers,
            "downsample_parameters": args.vqvae_downsample_parameters,
            "upsample_parameters": args.vqvae_upsample_parameters,
            "num_channels": args.vqvae_num_channels,
            "num_res_channels": args.vqvae_num_res_channels,
            "num_embeddings": args.vqvae_num_embeddings,
            "embedding_dim": args.vqvae_embedding_dim,
            "decay": args.vqvae_decay,
            "commitment_cost": args.vqvae_commitment_cost,
            "epsilon": args.vqvae_epsilon,
            "dropout": args.vqvae_dropout,
            "ddp_sync": args.vqvae_ddp_sync,
        }
        self.model = VQVAE(**vqvae_args)
        
        print(f"{sum(p.numel() for p in self.model.parameters()):,} VQVAE parameters")

        self.discriminator = PatchDiscriminator(
            spatial_dims=args.spatial_dimension,
            num_layers_d=3,
            num_channels=64,
            in_channels=args.vqvae_in_channels,
            out_channels=args.vqvae_out_channels,
            kernel_size=4,
            activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
            norm="BATCH",
            bias=False,
            padding=1,
        )
        self.discriminator.to(self.device)
        print(f"{sum(p.numel() for p in self.discriminator.parameters()):,} discriminator parameters")

        self.perceptual_loss = PerceptualLoss(
            spatial_dims=args.spatial_dimension, network_type="alex", is_fake_3d=True
        )
        self.perceptual_loss.to(self.device)
        self.jukebox_loss = JukeboxLoss(spatial_dims=args.spatial_dimension)
        self.jukebox_loss.to(self.device)
        self.optimizer_g = torch.optim.Adam(
            params=self.model.parameters(), lr=args.vqvae_learning_rate
        )
        self.optimizer_d = torch.optim.Adam(params=self.discriminator.parameters(), lr=5e-4)

        self.l1_loss = L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="hinge")
        self.adv_weight = args.adversarial_weight
        self.perceptual_weight = 0.001
        self.adversarial_warmup = bool(args.adversarial_warmup)
        # set up optimizer, loss, checkpoints
        os.makedirs(args.output_dir, exist_ok=True)
        self.run_dir = Path(args.output_dir) / args.model_name
        os.makedirs(self.run_dir, exist_ok=True)
        checkpoint_path = self.run_dir / "checkpoint.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.start_epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint["global_step"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
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
        with open(self.run_dir / "vqvae_config.json", "w") as f:
            json.dump(vqvae_args, f, indent=4)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=2.5e-5)
        if checkpoint_path.exists():
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # wrap the model with DistributedDataParallel module
        if self.ddp:
            print("Wrapping model with DistributedDataParallel.")
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
            self.discriminator = DistributedDataParallel(
                self.discriminator,
                device_ids=[self.device],
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        if args.quick_test:
            print("Quick test enabled, only running on a single train and eval batch.")
        self.quick_test = args.quick_test
        # self.logger_train = SummaryWriter(log_dir=str(self.run_dir / "train"))
        # self.logger_val = SummaryWriter(log_dir=str(self.run_dir / "val"))
        self.num_epochs = args.n_epochs

        train_dataset = BREASTDataset(root_dir=args.data_dir)
        self.train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        print("size of train dataset: ", len(train_dataset))
        valid_dataset = BREASTDataset(root_dir=args.data_dir_val)
        self.val_loader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        print("size of valid dataset: ", len(valid_dataset))
        wandb.init(project=args.project, group=args.model_name, config=args, entity="sinaamirrajab")



    def save_checkpoint(self, path, epoch, save_message=None):
        if self.ddp and dist.get_rank() == 0:
            # if DDP save a state dict that can be loaded by non-parallel models
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)
        if not self.ddp:
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
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
        wandb.finish()
        if self.ddp:
            dist.destroy_process_group()

    def train_epoch(self, epoch):
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            ncols=110,
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
            self.optimizer_g.zero_grad(set_to_none=True)

            # Generator part
            reconstruction, quantization_loss = self.model(images=images)
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]

            recons_loss = self.l1_loss(reconstruction.float(), images.float())
            perceptual_loss = self.perceptual_loss(reconstruction.float(), images.float())
            jukebox_loss = self.jukebox_loss(reconstruction.float(), images.float())
            adversarial_loss = self.adv_loss(
                logits_fake, target_is_real=True, for_discriminator=False
            )
            if self.adversarial_warmup:
                adv_weight = self.adv_weight * min(epoch, 50) / 50
            else:
                adv_weight = self.adv_weight
            total_generator_loss = (
                recons_loss
                + quantization_loss
                + self.perceptual_weight * perceptual_loss
                + jukebox_loss
                + adv_weight * adversarial_loss
            )

            total_generator_loss.backward()
            self.optimizer_g.step()

            # Discriminator part
            self.optimizer_d.zero_grad(set_to_none=True)

            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(images.contiguous().detach())[-1]
            loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            self.optimizer_d.step()

            l1_loss += recons_loss.item()
            generator_epoch_loss += total_generator_loss.item()
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
            wandb.log({
                    "recons_loss": recons_loss.item(),
                    "total_generator_loss": total_generator_loss.item(),
                    "gen_adv_loss": adversarial_loss.item(),
                    "disc_loss": discriminator_loss.item(),
                    "pereceptual_loss": perceptual_loss.item(),
                })

        
            if step%100== 0: # log to wandb
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                for i in range(2):
                    for j in range(4):
                        slice_idx = int(images.shape[-1]/2 + j*10)
                        image_to_show = images[i, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                        rec_to_show = reconstruction[i, :, :, :, slice_idx].squeeze().float().detach().cpu().numpy()
                        axes[i, j].imshow(np.concatenate([image_to_show, rec_to_show],axis=1), cmap='gray')      
                        axes[i, j].axis('off')
                    axes[i, j].set_title(batch['name'][i])
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
                plt.tight_layout()
                os.makedirs(os.path.join(self.run_dir, 'train_img'), exist_ok=True)
                file_path = os.path.join(self.run_dir, 'train_img', f"epoch_{epoch}_step_{step}.png")
                plt.savefig(file_path)
                plt.close(fig)
                # wandb.log({"training_image_recon": fig})

            if self.quick_test:
                break
        epoch_loss = generator_epoch_loss / epoch_step

        
        return epoch_loss

    def val_epoch(self, epoch):
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                ncols=110,
                position=0,
                leave=True,
                desc="Validation",
            )

            global_val_step = self.global_step
            for step, batch in progress_bar:
                images = batch["image"].to(self.device)
                reconstruction, quantization_loss = self.model(images=images)
                logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]

                recons_loss = self.l1_loss(reconstruction.float(), images.float())
                perceptual_loss = self.perceptual_loss(reconstruction.float(), images.float())
                jukebox_loss = self.jukebox_loss(reconstruction.float(), images.float())
                adversarial_loss = self.adv_loss(
                    logits_fake, target_is_real=True, for_discriminator=False
                )
                total_generator_loss = (
                    recons_loss
                    + quantization_loss
                    + self.perceptual_weight * perceptual_loss
                    + jukebox_loss
                    + self.adv_weight * adversarial_loss
                )
                wandb.log({
                            "validation_recons_loss": recons_loss.item(),
                            "validation_generator_loss": total_generator_loss.item(),
                        })

                global_val_step += images.shape[0]

                # plot some recons
                if step== 0: # log to wandb
                    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                    for i in range(2):
                        for j in range(4):
                            slice_idx = int(images.shape[-1]/2 + j*10)
                            image_to_show = images[i, :, :, :, slice_idx].squeeze().float().detach().cpu()
                            rec_to_show = reconstruction[i, :, :, :, slice_idx].squeeze().float().detach().cpu()
                            axes[i, j].imshow(np.concatenate([image_to_show, rec_to_show],axis=1), cmap='gray')      
                            axes[i, j].axis('off')
                        axes[i, j].set_title(batch['name'][i])
                    # Adjust layout to reduce empty spaces
                    plt.subplots_adjust(wspace=0.1, hspace=0.1)
                    plt.tight_layout()
                    plt.tight_layout()
                    os.makedirs(os.path.join(self.run_dir, 'valid_img'), exist_ok=True)
                    file_path = os.path.join(self.run_dir, 'valid_img', f"epoch_{epoch}_step_{step}.png")
                    plt.savefig(file_path)
                    plt.close(fig)
                    
                    # wandb.log({"validation_image_recon": fig})
