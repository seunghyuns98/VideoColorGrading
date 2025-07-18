import numpy as np
from PIL import Image
from grading import Inference
import argparse

def color_grading(args):
    grader = Inference(config=args.config)
    ref_path = args.ref_path
    input_path = args.input_path
    save_path = args.save_path

   
    reference_image = Image.open(ref_path).convert('RGB').resize((args.size, args.size))
    reference_image = np.array(reference_image)

    seed = args.seed
    steps = args.steps
    size = args.size

    output_path = grader(reference_image, input_path, save_path, seed, steps, size)
    print(f"Result saved at {output_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Animate images using given parameters.")
    parser.add_argument('--ref_path', type=str, required=True, help='Path to the reference images or videos.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input sequence file.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save output sequence file.')
    parser.add_argument('--config', type=str, default='configs/prompts/video_demo.yaml', help='Path to the configuration file.')
    parser.add_argument('--seed', type=int, help='Seed value.', default=26)
    parser.add_argument('--steps', type=int, help='Number of steps for the animation.', default=25)
    parser.add_argument('--size', type=int, help='Size of the image.', default=512)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    color_grading(args)
