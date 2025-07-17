import os
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
import random
import natsort 
import torch
from einops import rearrange
from transformers import CLIPProcessor
from pillow_lut import load_cube_file, resize_lut, identity_table

class Movie_3K_Step2(Dataset):
    def __init__(self,
                 video_folder,
                 sample_size=768,
                 clip_model_path="openai/clip-vit-base-patch32",
                 is_train=True
                 ):
        self.is_train = is_train
        self.spilt = 'train' if self.is_train else 'test'
        self.data_root = os.path.join(video_folder, self.spilt)
        self.mode = self.spilt
        self.dataset = natsort.natsorted([_ for _ in os.listdir(self.data_root)])

        self.num_videos = len(self.dataset)

        self.clip_image_processor = CLIPProcessor.from_pretrained(clip_model_path)
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize([sample_size[0], sample_size[0]]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.lut_root = 'PATH to Your LUT Files'
        LUTs = natsort.natsorted(
            [os.path.join(self.lut_root, _) for _ in os.listdir(self.lut_root) if _.endswith('.cube')])
        self.lut_loads = []

        for lut in LUTs:
            try:
                hefe = load_cube_file(lut)
            except Exception as e:
                continue
            hefe = resize_lut(hefe, 16)
            self.lut_loads.append(hefe)

        self.num_folders = len(self.lut_loads)
        self.length = self.num_videos
        self.identity_lut = identity_table(16).table.reshape(64, 64, 3)

    def __len__(self):
        return self.length

    def get_batch(self, v_idx, l_idx1, l_idx2, l_idx3):
        folder_name = self.dataset[v_idx]

        video_dir = os.path.join(self.data_root, folder_name)

        video_reader = natsort.natsorted([os.path.join(video_dir, _) for _ in os.listdir(video_dir) if _.endswith('.jpg')])
        video_length = len(video_reader)
        lut1 = self.lut_loads[l_idx1]
        lut1 = np.array(lut1.table, copy=False, dtype=np.float32)
        lut2 = self.lut_loads[l_idx2]
        lut2 = np.array(lut2.table, copy=False, dtype=np.float32)
        lut3 = self.lut_loads[l_idx3]
        lut3 = np.array(lut3.table, copy=False, dtype=np.float32)
        a = random.uniform(-1, 1)
        b = random.uniform(-1, 1)
        c = 1 - a - b
        lut = a*lut1 +b*lut2 +c*lut3
        lut = np.clip(lut, 0.0, 1.0)
        
        lut = ImageFilter.Color3DLUT(16, lut)

        batch_index = [random.randint(0, video_length - 1)]
        input_frames = []
        for batch in batch_index:
            image = Image.open(video_reader[batch]).convert('RGB')
            clip_input_frame = self.clip_image_processor(images=image, return_tensors="pt").pixel_values

            input_frame = torch.from_numpy(np.array(image)).unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            input_frame = input_frame / 255.
            input_frames.append(input_frame)

        input_frames = torch.cat(input_frames, dim=0)

        ref_img_idx = random.randint(0, video_length - 1)
        ref_frame = Image.open(video_reader[ref_img_idx]).convert('RGB')
        ref_frame = ref_frame.filter(lut)

        clip_ref_frame = self.clip_image_processor(images=ref_frame, return_tensors="pt").pixel_values

        ref_frames = torch.from_numpy(np.array(ref_frame)).permute(2, 0, 1).contiguous()
        ref_frames = ref_frames / 255.
        return input_frames, ref_frames, clip_input_frame, clip_ref_frame, lut

    def __getitem__(self, idx):
        v_idx = idx % self.num_videos
        l_idx1 = random.randint(0, self.num_folders-1)
        l_idx2 = random.randint(0, self.num_folders-1)
        l_idx3 = random.randint(0, self.num_folders-1)
        input_frames, ref_frame, clip_input_frame, clip_ref_frame, lut = self.get_batch(v_idx, l_idx1, l_idx2, l_idx3)
        
        lut = np.array(lut.table, copy=False, dtype=np.float32)
        lut = lut.reshape(64,64,3)
        lut = lut - self.identity_lut
        lut = rearrange(lut, "h w c -> c h w")
        identity_lut = rearrange(self.identity_lut, "h w c -> c h w")
        input_frames = self.pixel_transforms(input_frames)
        ref_frame = ref_frame.unsqueeze(0)
        ref_frame = self.pixel_transforms(ref_frame)

        drop_image_embeds = 1 if random.random() < 0.1 else 0
        sample = dict(
            input_frames=input_frames,
            ref_frame=ref_frame,
            clip_input_frame=clip_input_frame,
            clip_ref_frame=clip_ref_frame,
            lut=lut,
            id_lut=identity_lut,
            drop_image_embeds=drop_image_embeds,
        )

        return sample


