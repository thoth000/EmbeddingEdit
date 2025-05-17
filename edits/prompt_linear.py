import torch
import os
import argparse
from diffusers import StableDiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from models import embed2img
from tqdm import tqdm

def linear_interpolate_prompt_embeddings(pipe, prompt_A, prompt_B, negative_prompt, num_interps, out_dir, seed=1234):
    emb_A = embed2img.get_text_emb(pipe, prompt_A)
    emb_B = embed2img.get_text_emb(pipe, prompt_B)
    neg_emb = embed2img.get_text_emb(pipe, negative_prompt)

    for i, alpha in tqdm(enumerate(torch.linspace(0, 1, steps=num_interps)), total=num_interps):
        emb_interp = (1 - alpha) * emb_A + alpha * emb_B # linear interpolation
        latents = embed2img.generate_latents_from_embedding(pipe, emb_interp, negative_prompt_embeds=neg_emb, seed=seed)
        save_path = os.path.join(out_dir, f"interp_{i}.png")
        embed2img.decode_and_save(pipe, latents, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear interpolate between two prompt embeddings and generate images.")
    parser.add_argument("--prompt_a", type=str, required=True)
    parser.add_argument("--prompt_b", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="ugly, blurry, low quality") # ネガティブプロンプト引数を追加
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    # 出力先ディレクトリの解決
    if args.out_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        args.out_dir = os.path.join(base_dir, "outputs", "linear_interpolated")

    os.makedirs(args.out_dir, exist_ok=True)

    # パイプラインの読み込みと補間実行
    model_id = "stabilityai/stable-diffusion-2-1-base"
    vae_model_id = "stabilityai/sd-vae-ft-mse"
    
    vae = AutoencoderKL.from_pretrained(vae_model_id)
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        vae=vae,
    )
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe = pipe.to("cuda")

    linear_interpolate_prompt_embeddings(
        pipe,
        prompt_A=args.prompt_a,
        prompt_B=args.prompt_b,
        negative_prompt=args.negative_prompt,
        num_interps=args.steps,
        out_dir=args.out_dir,
        seed=args.seed
    )
