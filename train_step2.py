import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from typing import Dict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from data.dataset import collate_fn_2
from utils.util import zero_rank_print
from models.ReferenceEncoder import ReferenceEncoder
from models.ReferenceNet import ReferenceNet
from models.ReferenceNet_attention_diff import ReferenceNetAttention
from data.Movie_3K_Step2 import Movie_3K_Step2

def init_dist(launcher="slurm", backend='nccl', port=28888, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank


def get_parameters_without_gradients(model):
    """
    Returns a list of names of the model parameters that have no gradients.

    Args:
    model (torch.nn.Module): The model to check.
    
    Returns:
    List[str]: A list of parameter names without gradients.
    """
    no_grad_params = []
    for name, param in model.named_parameters():
        print(f"{name} : {param.grad}")
        if param.grad is None:
            no_grad_params.append(name)
    return no_grad_params


def main(
    name: str,
    use_wandb: bool,
    launcher: str,
    
    output_dir: str,
    pretrained_model_path: str,
    clip_model_path:str,
    step1_checkpoint_path: str,

    description: str,
    fusion_blocks: str,
    train_data: Dict,
    
    noise_scheduler_kwargs = None,
    
    max_train_steps: int = 100,

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 100,
    lr_scheduler: str = "linear",

    num_workers: int = 16,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
):
    check_min_version("0.21.4")

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher, port=28887)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)
    

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="Video Color Grading train stage 2", name=folder_name, config=config)



    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
        
        print(description)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    clip_image_encoder = ReferenceEncoder(model_path=clip_model_path)
    referencenet = ReferenceNet.from_pretrained(pretrained_model_path, subfolder="unet")

    state_dict = torch.load(step1_checkpoint_path, map_location="cpu")
    referencenet_state_dict = state_dict['referencenet_state_dict']
    referencenet.load_state_dict(referencenet_state_dict, strict=True)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", in_channels=6, out_channels=3, cross_attention_dim=1280, up_block_types= ["UpBlock2D","UpBlock2D","UpBlock2D", "UpBlock2D"], down_block_types= ["DownBlock2D","DownBlock2D", "DownBlock2D", "DownBlock2D"], low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
        
    reference_control_writer = ReferenceNetAttention(referencenet, do_classifier_free_guidance=False, mode='write', fusion_blocks=fusion_blocks)
    reference_control_reader = ReferenceNetAttention(unet, do_classifier_free_guidance=False, mode='read', fusion_blocks=fusion_blocks)
          
    vae.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)
    unet.requires_grad_(True)
    referencenet.requires_grad_(False)    
                   
    
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    print(f"trainable params number: {len(trainable_params)}")
    print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            referencenet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        referencenet.enable_gradient_checkpointing()

    vae.to(local_rank)
    clip_image_encoder.to(local_rank)
    referencenet.to(local_rank)

    train_dataset = Movie_3K_Step2(**train_data)
    
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_2,
    )

    # Get the training iteration

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # DDP warpper
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None
    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        referencenet.train()
        
        for step, batch in enumerate(train_dataloader):

            ### >>>> Training >>>> ###
            
            # Convert videos to latent space            
            input_frames = batch["input_frames"].to(local_rank)
            ref_frame = batch["ref_frame"].to(local_rank)
            clip_input_frame = batch["clip_input_frame"].to(local_rank)
            clip_ref_frame = batch["clip_ref_frame"].to(local_rank)
            lut = batch["lut"].to(local_rank)
            id_lut = batch["id_lut"].to(local_rank)
            drop_image_embeds = batch["drop_image_embeds"].to(local_rank) # torch.Size([bs])
            
            with torch.no_grad():
                ref_latents = vae.encode(ref_frame).latent_dist
                ref_latents = ref_latents.sample()
                ref_latents = ref_latents * 0.18215

                src_latents = vae.encode(input_frames).latent_dist
                src_latents = src_latents.sample()
                src_latents = src_latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(lut)
            bsz = ref_latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=ref_latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_lut = noise_scheduler.add_noise(lut, noise, timesteps)
            
            
            noisy_lut = torch.cat([noisy_lut, id_lut], dim=1)
            
            # Get the text embedding for conditioning
            with torch.no_grad():
                clip_input = torch.cat([clip_ref_frame, clip_input_frame])
                encoder_hidden_states = clip_image_encoder(clip_input).unsqueeze(1) # [bs,1,768]
            mask = drop_image_embeds > 0
            mask1 = mask.unsqueeze(1).unsqueeze(2).repeat(2, 1, 1).expand_as(encoder_hidden_states)
            encoder_hidden_states[mask1] = 0
            
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                ref_timesteps = torch.zeros_like(timesteps).repeat(2)
                
                # pdb.set_trace()
                ref_input = torch.cat([ref_latents, src_latents], dim=0)
                referencenet(ref_input, ref_timesteps, encoder_hidden_states)
                reference_control_reader.update(reference_control_writer)
                model_pred = unet(noisy_lut, timesteps, None).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                
            optimizer.zero_grad()

            loss.backward()
                
            """ >>> gradient clipping >>> """
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            """ <<< gradient clipping <<< """
            optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            
            reference_control_reader.clear()
            reference_control_writer.clear()
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if is_main_process and global_step % checkpointing_steps == 0 :
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "unet_state_dict": unet.module.state_dict(),
                }
                if step == len(train_dataloader) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-global_step-{global_step}.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
    