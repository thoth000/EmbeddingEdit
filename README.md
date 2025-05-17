# EmbeddingEdit
![sample image](https://github.com/thoth000/EmbeddingEdit/blob/main/docs/images/image.png)

## Features
- I implement each step of image generation, so interpolation is possible at any stage.
  1. prompt tokenization
  2. raw word embedding
  3. prompt embedding
  4. denozing

- image generation from a prompt
- images generation along with { `linear`, ... }interpolation between { `prompt`, } embedding

## Tutorial
1. You have to prepare the environment.
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install diffusers["torch"] transformers
   pip install accelerate
   ```
2. You can run programs.
   - You are available to generate image from a prompt.
     ```bash
     python single.py --prompt "a dog playing with ball"
     ```
   - You are also available to generate few images along with linear interpolation.
     ```bash
     python interp.py --prompt_a "Only an apple" --prompt_b "ten apples"
     ```
