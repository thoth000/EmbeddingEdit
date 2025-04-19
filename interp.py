import os
import argparse
from diffusers import StableDiffusionPipeline
from edits import prompt_linear

"""
Sample usage:
python interp.py --prompt_a "a red apple" --prompt_b "a green apple" --out_dir outputs/apple_interp
"""

def main():
    parser = argparse.ArgumentParser(description="Run prompt embedding interpolation.")
    parser.add_argument("--prompt_a", type=str, required=True, help="Starting prompt")
    parser.add_argument("--prompt_b", type=str, required=True, help="Ending prompt")
    parser.add_argument("--steps", type=int, default=5, help="Number of interpolation steps")
    parser.add_argument("--out_dir", type=str, default="outputs/interpolated", help="Output directory")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for latent noise")
    parser.add_argument("--edit_type", type=str, default="prompt_linear", choices=["prompt_linear"], help="Type of edits to apply")

    args = parser.parse_args()

    # 出力先ディレクトリの作成（責任は main 側に）
    os.makedirs(args.out_dir, exist_ok=True)

    # Stable Diffusion パイプラインの読み込み
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
    ).to("cuda")
    
    if args.edit_type == "prompt_linear":
        # prompt_linear モジュールを使用して補間
        prompt_linear.linear_interpolate_prompt_embeddings(
            pipe=pipe,
            prompt_A=args.prompt_a,
            prompt_B=args.prompt_b,
            num_interps=args.steps,
            out_dir=args.out_dir,
            seed=args.seed
        )
    else:
        raise ValueError(f"Unknown edits type: {args.edits_type}")

if __name__ == "__main__":
    main()
