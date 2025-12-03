#@title Utils Code
%cd /content/ComfyUI

import os, random, time

import torch
import numpy as np
from PIL import Image
import re, uuid
from nodes import NODE_CLASS_MAPPINGS

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

with torch.inference_mode():
    unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
    vae = VAELoader.load_vae("ae.safetensors")[0]

save_dir="./results"
os.makedirs(save_dir, exist_ok=True)
def get_save_path(prompt):
  save_dir = "./results"
  safe_prompt = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:25]
  uid = uuid.uuid4().hex[:6]
  filename = f"{safe_prompt}_{uid}.png"
  path = os.path.join(save_dir, filename)
  return path

@torch.inference_mode()
def generate(input):
    values = input["input"]
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    seed = values['seed'] # 0
    steps = values['steps'] # 9
    cfg = values['cfg'] # 1.0
    sampler_name = values['sampler_name'] # euler
    scheduler = values['scheduler'] # simple
    denoise = values['denoise'] # 1.0
    width = values['width'] # 1024
    height = values['height'] # 1024
    batch_size = values['batch_size'] # 1.0

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0]
    samples = KSampler.sample(unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    save_path=get_save_path(positive_prompt)
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(save_path)
    return save_path,seed
    
import gradio as gr
def generate_ui(
    positive_prompt,
    negative_prompt,
    aspect_ratio,
    seed,
    steps,
    cfg,
    denoise,
    batch_size=1,
    sampler_name="euler",
    scheduler="simple"
):
    width, height = [int(x) for x in aspect_ratio.split("(")[0].strip().split("x")]

    input_data = {
        "input": {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": int(batch_size),
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": float(denoise),
        }
    }

    image_path,seed = generate(input_data)
    return image_path,image_path,seed


# ---------------- DEFAULT PROMPTS ----------------

DEFAULT_POSITIVE = """A photorealistic tabby cat sprinting through a Dutch fish market
while hugging a giant silver fish, dramatic face, cinematic lighting, freeze-frame action,
shallow depth of field, ultra-detailed, natural motion blur."""

DEFAULT_NEGATIVE = """low quality, blurry, unnatural skin tone, bad lighting, pixelated,
noise, oversharpen, soft focus,pixelated"""

ASPECTS = [
    "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "864x1152 (3:4)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", "720x1280 (9:16)",
    "1344x576 (21:9)", "576x1344 (9:21)"
]


custom_css = ".gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"

with gr.Blocks(theme=gr.themes.Soft(),css=custom_css) as demo:
  gr.HTML("""
<div style=\"width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; margin:20px 0;\">
    <h1 style=\"font-size:2.5em; margin-bottom:10px;\">Z-Image-Turbo</h1>
    <a href=\"https://github.com/Tongyi-MAI/Z-Image\" target=\"_blank\">
        <img src=\"https://img.shields.io/badge/GitHub-Z--Image-181717?logo=github&logoColor=white\"
             style=\"height:15px;\">
    </a>
</div>
""")


  with gr.Row():
    with gr.Column():
      positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=5)

      with gr.Row():
        aspect = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="Aspect Ratio")
        seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
        steps = gr.Slider(4, 25, value=9, step=1, label="Steps")
      with gr.Row():
        run = gr.Button('🚀 Generate', variant='primary')
      with gr.Accordion('Image Settings', open=False):
        with gr.Row():
          cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
          denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
        with gr.Row():
          negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3)
    with gr.Column():
        download_image=gr.File(label="Download Image")
        output_img = gr.Image(label="Generated Image", height=480)
        used_seed = gr.Textbox(label="Seed Used", interactive=False,show_copy_button=True)

    run.click(
        fn=generate_ui,
        inputs=[positive, negative, aspect, seed, steps, cfg, denoise,],
        outputs=[download_image,output_img, used_seed]
    )

demo.launch(share=True, debug=True)
