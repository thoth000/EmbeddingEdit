import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from edits import prompt_linear

"""
Sample usage:
python interp.py --prompt_a "Red sedan car, front view, driving down the center of a wide, empty highway." --prompt_b "Green sedan car, front view, driving down the center of a wide, empty highway." --steps 10
"""

def main():
    parser = argparse.ArgumentParser(description="Run prompt embedding interpolation.")
    parser.add_argument("--prompt_a", type=str, required=True, help="Starting prompt")
    parser.add_argument("--prompt_b", type=str, required=True, help="Ending prompt")
    parser.add_argument("--negative_prompt", type=str, default="ugly, blurry, low quality")
    parser.add_argument("--steps", type=int, default=5, help="Number of interpolation steps")
    parser.add_argument("--out_dir", type=str, default="outputs/interpolated", help="Output directory")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for latent noise")
    parser.add_argument("--edit_type", type=str, default="prompt_linear", choices=["prompt_linear"], help="Type of edits to apply")

    args = parser.parse_args()

    # 出力先ディレクトリの作成（責任は main 側に）
    os.makedirs(args.out_dir, exist_ok=True)

    # Stable Diffusion パイプライン
    model_id = "stabilityai/stable-diffusion-2-1-base"
    vae_model_id = "stabilityai/sd-vae-ft-mse"
    
    vae = AutoencoderKL.from_pretrained(vae_model_id)
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        vae=vae,
    )
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe = pipe.to("cuda")
    
    if args.edit_type == "prompt_linear":
        # prompt_linear モジュールを使用して補間
        prompt_linear.linear_interpolate_prompt_embeddings(
            pipe=pipe,
            prompt_A=args.prompt_a,
            prompt_B=args.prompt_b,
            negative_prompt=args.negative_prompt,
            num_interps=args.steps,
            out_dir=args.out_dir,
            seed=args.seed
        )
    else:
        raise ValueError(f"Unknown edits type: {args.edit_type}")

if __name__ == "__main__":
    main()
