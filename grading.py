import inspect
import os
import numpy as np
from PIL import Image, ImageFilter

from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from tqdm import tqdm
from transformers import CLIPProcessor

from models.ImageEncoder import ImageEncoder
from models.ReferenceNet import ReferenceNet

from pipeline import InferencePipeline
from diffusers.models import UNet2DConditionModel

from utils.util import save_videos_grid
from utils.videoreader import VideoReader

from accelerate.utils import set_seed
from einops import rearrange
from pillow_lut import identity_table

class Inference():
    def __init__(self, config="configs/prompts/video_demo.yaml") -> None:
        print("Initializing LUT Generation Pipeline...")
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)
        config  = OmegaConf.load(config)  
        inference_config = OmegaConf.load(config.inference_config)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ### >>> create diffusion pipeline >>> ###
        vae = AutoencoderKL.from_pretrained(config.pretrained_sd_path, subfolder="vae")
        self.clip_image_encoder = ImageEncoder(model_path=config.pretrained_clip_path)
        self.clip_image_processor = CLIPProcessor.from_pretrained(config.pretrained_clip_path, local_files_only=True)

        unet = UNet2DConditionModel.from_pretrained(config.pretrained_sd_path, subfolder="unet", in_channels=6, out_channels=3, cross_attention_dim=1280, up_block_types= ["UpBlock2D","UpBlock2D","UpBlock2D", "UpBlock2D"], down_block_types= ["DownBlock2D","DownBlock2D", "DownBlock2D", "DownBlock2D"], low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
        state_dict = torch.load(config.pretrained_LD_path, map_location="cpu")['unet_state_dict']
        m, u = unet.load_state_dict(state_dict, strict=True)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)} ###")
        if len(m) !=0 or len(u) !=0:
            print(f"### missing keys:\n{m}\n### unexpected keys:\n{u}\n ###")
        
        self.referencenet = ReferenceNet.from_pretrained(config.pretrained_sd_path, subfolder="unet")
        state_dict = torch.load(config.pretrained_GE_path, map_location="cpu")["referencenet_state_dict"]
        m, u = self.referencenet.load_state_dict(state_dict, strict=True)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)} ###")
        if len(m) !=0 or len(u) !=0:
            print(f"### missing keys:\n{m}\n### unexpected keys:\n{u}\n ###")

        self.id_lut_hwc = identity_table(16).table.reshape(64, 64, 3)
        self.id_lut_chw = torch.from_numpy(rearrange(self.id_lut_hwc, "h w c -> c h w")).unsqueeze(0)

        self.pipeline = InferencePipeline(
            vae=vae, unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        )
        self.pipeline.to(device)
        print("Initialization Done!")

    def __call__(self, ref_sequence, input_path, save_path, random_seed, step, size=512):
       
        input_video = VideoReader(input_path).read()
        if input_video[0].shape[0] != size:
            input_video = [np.array(Image.fromarray(c)) for c in input_video][:480]
            input_video_resize = [np.array(Image.fromarray(c).resize((size, size))) for c in input_video]
        input_video_resize = np.array(input_video_resize)
        input_video = np.array(input_video)

        random_seed = int(random_seed)
        step = int(step)

        torch.manual_seed(random_seed)
        set_seed(random_seed)

        generator = torch.Generator(device=torch.device("cuda:0"))
        generator.manual_seed(torch.initial_seed())
    
        
        lut = self.pipeline(
            num_inference_steps      = step,
            width                    = size,
            height                   = size,
            generator                = generator,
            num_actual_inference_steps = step,
            source_image             = ref_sequence,
            referencenet             = self.referencenet,
            clip_image_processor     = self.clip_image_processor,
            clip_image_encoder       = self.clip_image_encoder,
            input_video              = input_video_resize,
            id_lut                   = self.id_lut_chw,
            return_dict = False
        )
    
        lut = lut[0].detach().cpu().numpy()
        lut = rearrange(lut, "c h w -> h w c")
        lut = lut + self.id_lut_hwc
        lut = np.clip(lut, 0.0, 1.0)
        lut = lut.flatten()
        lut = ImageFilter.Color3DLUT(16, lut)
        output_frames = []

        for frame in tqdm(input_video):
            output_frame = np.array(Image.fromarray(frame).filter(lut))/ 255.0
            output_frames.append(output_frame)
        output_frames = np.array(output_frames) 
        output_frames = rearrange(output_frames, "t h w c -> 1 c t h w")
        output_video = torch.from_numpy(output_frames)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_videos_grid(output_video, save_path)
        
        return save_path
