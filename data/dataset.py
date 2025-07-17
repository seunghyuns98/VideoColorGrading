import torch
import torch.distributed as dist
def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


def collate_fn(data):

    target_frames = torch.cat([example["target_frames"] for example in data], dim=0)
    input_frames = torch.cat([example["input_frames"] for example in data], dim=0)
    clip_ref_frame = torch.cat([example["clip_ref_frame"] for example in data], dim=0)
    ref_frame = torch.cat([example["ref_frame"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embeds"] for example in data]
    drop_image_embeds = torch.Tensor(drop_image_embeds)
    return {
        "target_frames": target_frames,
        "input_frames": input_frames,
        "clip_ref_frame": clip_ref_frame,
        "ref_frame": ref_frame,
        "drop_image_embeds": drop_image_embeds,
    }

def collate_fn_2(data):
   
    input_frames = torch.cat([example["input_frames"] for example in data], dim=0)
    ref_frame = torch.cat([example["ref_frame"] for example in data], dim=0)
    clip_input_frame = torch.cat([example["clip_input_frame"] for example in data], dim=0)
    clip_ref_frame = torch.cat([example["clip_ref_frame"] for example in data], dim=0)
    lut = torch.stack([torch.from_numpy(example["lut"]) for example in data])
    id_lut = torch.stack([torch.from_numpy(example["id_lut"]) for example in data])
    drop_image_embeds = [example["drop_image_embeds"] for example in data]
    drop_image_embeds = torch.Tensor(drop_image_embeds)
    
    return {
        "input_frames": input_frames,
        "ref_frame": ref_frame,
        "clip_input_frame": clip_input_frame,
        "clip_ref_frame": clip_ref_frame,
        "lut": lut,
        "id_lut": id_lut,
        "drop_image_embeds": drop_image_embeds,
    }
