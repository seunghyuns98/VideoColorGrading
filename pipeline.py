import inspect, math
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from packaging import version
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange
from models.ReferenceNet_attention_diff import ReferenceNetAttention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class InferencePipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class InferencePipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: None,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device


    def prepare_extra_step_kwargs(self, generator, eta):

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        rand_device = "cpu" if device.type == "mps" else device

        if isinstance(generator, list):
            latents = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                for i in range(batch_size)
            ]
            latents = torch.cat(latents, dim=0).to(device)
        else:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
                
        return latents
    
    @torch.no_grad()
    def images2latents(self, images, dtype):
        """
        Convert RGB image to VAE latents
        """
        device = self._execution_device
        images = torch.from_numpy(images).float().to(dtype) / 127.5 - 1
        images = rearrange(images, "f h w c -> f c h w").to(device)
        latents = []
        for frame_idx in range(images.shape[0]):
            latents.append(self.vae.encode(images[frame_idx:frame_idx+1])['latent_dist'].mean * 0.18215)
        latents = torch.cat(latents)
        return latents
    
    
    @torch.no_grad()
    def __call__(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 200,
        eta: float = 0.0,

        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True,
        num_actual_inference_steps: Optional[int] = None,
        id_lut: Optional[torch.FloatTensor] = None,
        referencenet = None,
        clip_image_processor = None,
        clip_image_encoder = None,
        input_video = None,
        source_image: str = None,
        **kwargs,
    ):

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        batch_size = 1

        device = self._execution_device

        reference_control_writer = ReferenceNetAttention(referencenet, mode='write', fusion_blocks="full")
        reference_control_reader = ReferenceNetAttention(self.unet, mode='read', fusion_blocks="full")
        
        is_dist_initialized = kwargs.get("dist", False)
        rank = kwargs.get("rank", 0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        embedding_dtype = torch.float32

        latents = self.prepare_latents(
            batch_size,
            3,
            height,
            width,
            embedding_dtype,
            device,
            generator,
        )


        latents_dtype = latents.dtype
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # For img2img setting
        if num_actual_inference_steps is None:
            num_actual_inference_steps = num_inference_steps
        clip_image_encoder.to(device=latents.device)
        
        cos_sim = torch.nn.CosineSimilarity()
        max_sim = 0
        source_image = np.expand_dims(source_image, axis=0)
        for idx_c in tqdm(range(0, len(input_video), 24)):
            clip_src_image = clip_image_processor(images=Image.fromarray(input_video[idx_c]).convert('RGB'), return_tensors="pt").pixel_values
            clip_src_vector = clip_image_encoder(clip_src_image.to(device=latents.device,dtype=latents.dtype)).unsqueeze(1).to(device=latents.device,dtype=latents.dtype)
            clip_ref_image = clip_image_processor(images=Image.fromarray(source_image[0]).convert('RGB'), return_tensors="pt").pixel_values
            clip_ref_vector = clip_image_encoder(clip_ref_image.to(device=latents.device,dtype=latents.dtype)).unsqueeze(1).to(device=latents.device,dtype=latents.dtype)
            sim_val = cos_sim(clip_ref_vector.squeeze(1), clip_src_vector.squeeze(1))
            if sim_val > max_sim:
                max_sim = sim_val
                src_img = input_video[idx_c]
                clip_src = clip_src_vector
                ref_img = source_image[0]
                clip_ref = clip_ref_vector
        src_image_latents = self.images2latents(src_img[None, :], latents_dtype).cuda()
        ref_image_latents = self.images2latents(ref_img[None, :], latents_dtype).cuda()
        encoder_hidden_states = torch.cat([clip_ref, clip_src], dim=0) # [bs,1,768]

        clip_image_encoder.to('cpu')
       
        
        batch_size = 1
        
        ref_input = torch.cat([ref_image_latents, src_image_latents], dim=0)
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=(rank!=0)):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue

            if i == 0:
                # just once
                referencenet.to(device=latents.device)
                referencenet(
                    ref_input,
                    torch.zeros_like(t),
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )
                reference_control_reader.update(reference_control_writer)
                referencenet.to('cpu')
            
       
            latent_model_input = latents.to(device)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            latents_pose_input = id_lut.to(device)
            

            noisy_latents = torch.cat([latent_model_input, latents_pose_input], dim=1)
            pred = self.unet(
                    noisy_latents, 
                    t, 
                    encoder_hidden_states=None,
                    return_dict=False,
                )[0]
            
            latents = self.scheduler.step(pred, t, latents, **extra_step_kwargs).prev_sample
            if is_dist_initialized:
                dist.broadcast(latents, 0)
                dist.barrier()
        if is_dist_initialized:
            dist.barrier()

        if not return_dict:
            return latents
        
        return AnimationPipelineOutput(videos=latents)
