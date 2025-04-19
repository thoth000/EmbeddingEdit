import argparse
import os
from diffusers import StableDiffusionPipeline
from models import embed2img


def main():
    parser = argparse.ArgumentParser(description="Generate a single image from one prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--out_path", type=str, default="outputs/image.png", help="Output file path")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = parser.parse_args()

    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    # モデル読み込み
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
    ).to("cuda")

    # 埋め込み取得
    embedding = embed2img.get_text_emb(pipe, args.prompt)

    # 潜在表現生成
    latents = embed2img.generate_latents_from_embedding(
        pipe,
        embedding,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed
    )

    # デコード & 保存
    embed2img.decode_and_save(pipe, latents, args.out_path)


if __name__ == "__main__":
    main()
