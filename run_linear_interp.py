import os
import argparse
from diffusers import StableDiffusionPipeline
from edits import prompt_linear


def main():
    parser = argparse.ArgumentParser(description="Run prompt embedding interpolation.")
    parser.add_argument("--prompt_a", type=str, required=True, help="Starting prompt")
    parser.add_argument("--prompt_b", type=str, required=True, help="Ending prompt")
    parser.add_argument("--steps", type=int, default=5, help="Number of interpolation steps")
    parser.add_argument("--out_dir", type=str, default="outputs/interpolated", help="Output directory")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for latent noise")

    args = parser.parse_args()

    # 出力先ディレクトリの作成（責任は main 側に）
    os.makedirs(args.out_dir, exist_ok=True)

    # Stable Diffusion パイプラインの読み込み
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype="auto"
    ).to("cuda")

    # 線形補間＆画像生成
    prompt_linear.linear_interpolate_prompt_embeddings(
        pipe=pipe,
        prompt_A=args.prompt_a,
        prompt_B=args.prompt_b,
        num_interps=args.steps,
        out_dir=args.out_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
