import os
import numpy as np
from PIL import Image
import argparse
from natsort import natsort
from pillow_lut import load_cube_file, resize_lut
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from utils.videoreader import VideoReader
from utils.util import save_videos_grid

    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Animate images using given parameters.")
    parser.add_argument('--video_root', type=str, required=True, help='Path to the motion sequence file.')
    parser.add_argument('--text_prompt', type=str, required=True, help='Text Prompt')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    lut_root = "Path To Your LUT Files"
    lut_text_path = "Path To Your LUT Texts"
    pretrained_clip_path = "Path To Your CLIP Pre-trained Wieghts "

    LUTs = natsort.natsorted([os.path.join(lut_root, _) for _ in os.listdir(lut_root) if _.endswith('.cube')])
    texts = open(lut_text_path, 'r')
    
    lut_loads = []
    lut_text_list = []
    lines = texts.readlines()
    count = 0
    for i, lut in enumerate(LUTs):
        line = lines[i-count].rstrip()

        name = lut.split('/')[-1]
        try:
            hefe = load_cube_file(lut)
        except Exception as e:
            count+=1
            continue
        hefe = resize_lut(hefe, 16)
        lut_loads.append(hefe)
        lut_text_list.append(line)
    
    text = args.text_prompt
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_clip_path)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_clip_path)  
    token = tokenizer(lut_text_list, padding=True, return_tensors="pt")
    token_text = tokenizer(text, padding=True, return_tensors="pt")
    t_embedding = text_encoder(**token).pooler_output
    t_embedding_t = text_encoder(**token_text).pooler_output
    cos_sim = torch.nn.CosineSimilarity()
    cos = cos_sim(t_embedding, t_embedding_t)
    idx = torch.argmax(cos)

    selected_lut = lut_loads[idx]

    video = VideoReader(args.video_root).read()
    frames = []
    for frame in video:
        frame_l = np.array(Image.fromarray(frame).filter(selected_lut))/ 255.0
        frames.append(frame_l)

    frames = np.array(frames) 
    frames = rearrange(frames, "t h w c -> 1 c t h w")
    frames = torch.from_numpy(frames)

    savedir = f"./test" 
    os.makedirs(savedir, exist_ok=True)

    animation_path = f"{savedir}/{args.video_root.split('/')[-1].split('.')[0]}.mp4"

    save_videos_grid(frames, animation_path)