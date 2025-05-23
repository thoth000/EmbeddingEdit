import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np


# プロンプト埋め込みを取得
def get_text_emb(pipe, prompt):
    tok = pipe.tokenizer(
        [prompt], padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)
    with torch.no_grad():
        return pipe.text_encoder(**tok).last_hidden_state


# 無条件＋条件付きを２倍バッチ化する関数
def make_cfg_batch(pipe, emb, negative_emb=None):
    if negative_emb is None:
        empty = get_text_emb(pipe, "")  # 無条件用 (従来通り)
    else:
        empty = negative_emb # 指定されたネガティブ埋め込みを使用
    return torch.cat([empty, emb], dim=0)


# デコード＆保存
def decode_and_save(pipe, latents, name):
    with torch.no_grad():
        img = pipe.vae.decode(latents[:1] / pipe.vae.config.scaling_factor).sample
    img = (img.clamp(-1, 1) + 1) / 2
    arr = (img.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)[0]
    Image.fromarray(arr).save(name)


# ノイズ生成 → デノイジング
def generate_latents_from_embedding(
    pipe,
    embedding: torch.Tensor,
    negative_prompt_embeds: torch.Tensor = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 1234,
):
    device = pipe.device
    pipe.scheduler.set_timesteps(num_inference_steps)

    gen = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        1, pipe.unet.config.in_channels,
        pipe.unet.config.sample_size,
        pipe.unet.config.sample_size,
        generator=gen, device=device
    ) * pipe.scheduler.init_noise_sigma

    emb_batch = make_cfg_batch(pipe, embedding, negative_prompt_embeds)

    for t in pipe.scheduler.timesteps:
        latents_pair = torch.cat([latents, latents], dim=0)
        with torch.no_grad():
            noise_pred = pipe.unet(latents_pair, t, encoder_hidden_states=emb_batch).sample
        uncond, cond = noise_pred.chunk(2)
        guided = uncond + guidance_scale * (cond - uncond)
        latents = pipe.scheduler.step(guided, t, latents).prev_sample

    return latents


if __name__ == "__main__":
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

    # エンベディングを生成
    prompt = "a red apple"
    emb = get_text_emb(pipe, prompt)
    # 潜在ベクトルを生成
    latents = generate_latents_from_embedding(pipe, emb)
    # デコード＆保存
    decode_and_save(pipe, latents, "apple.png")