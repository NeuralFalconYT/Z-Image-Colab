
# root_path = "/kaggle/working" # @param ['/content', '/root', '/kaggle/working','/teamspace/studios/this_studio']
# %cd $root_path/ComfyUI

import os 
root_path = os.path.dirname(os.getcwd())

import os, random, time, sys, re, uuid, gc, ctypes
import torch
import numpy as np
import importlib
import importlib.util
from PIL import Image, ImageFilter
import gradio as gr
import psutil


comfyui_path = f"{root_path}/ComfyUI"   
if comfyui_path in sys.path: sys.path.remove(comfyui_path)
sys.path.insert(0, comfyui_path)

for key in list(sys.modules.keys()):
    if key == 'utils' or key.startswith('utils.'):
        del sys.modules[key]

importlib.invalidate_caches()
comfy_utils_path = os.path.join(comfyui_path, "utils", "__init__.py")
if os.path.exists(comfy_utils_path):
    spec = importlib.util.spec_from_file_location("utils", comfy_utils_path, submodule_search_locations=[os.path.join(comfyui_path, "utils")])
    if spec and spec.loader:
        utils_module = importlib.util.module_from_spec(spec)
        sys.modules["utils"] = utils_module
        spec.loader.exec_module(utils_module)

import comfy.model_management


import nest_asyncio
nest_asyncio.apply()

import asyncio
import server
import execution
import nodes

print("Initializing ComfyUI Nodes...")
loop = asyncio.get_event_loop()
server_instance = server.PromptServer(loop)
execution.PromptQueue(server_instance)

init_result = nodes.init_extra_nodes()
if asyncio.iscoroutine(init_result):
    loop.run_until_complete(init_result)

from nodes import NODE_CLASS_MAPPINGS

# Initialize Node Classes
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
InpaintModelConditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
ModelSamplingAuraFlow = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()


# ======================================================================
# MEMORY MONITORING & GPU DETECTION 
# ======================================================================
def print_mem_stats(tag=""):
    sys_ram = psutil.virtual_memory().used / 1024**3
    mem_str = f"[DEBUG-MEM] [{tag}] Sys RAM: {sys_ram:.2f}GB"
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            vram_alloc = torch.cuda.memory_allocated(i) / 1024**3
            mem_str += f" | GPU {i} Alloc: {vram_alloc:.2f}GB"
    print(mem_str)

def gpu_vram_gb():
    if not torch.cuda.is_available(): return 0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

LOW_GPU = gpu_vram_gb() <= 15
DUAL_GPU = torch.cuda.is_available() and torch.cuda.device_count() >= 2
_Z_IMAGE_MODELS = None

def aggressive_clean():
    """Forces extreme garbage collection for both Python and OS RAM."""
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(f"cuda:{i}"):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass

def load_z_models():
    """Only used for HIGH VRAM or DUAL GPU systems."""
    global _Z_IMAGE_MODELS
    if LOW_GPU and not DUAL_GPU:
        return (None, None, None) 
        
    if _Z_IMAGE_MODELS is not None:
        return _Z_IMAGE_MODELS

    with torch.inference_mode():
        device_0 = torch.device("cuda:0")
        device_1 = torch.device("cuda:1") if DUAL_GPU else device_0
        
        unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
        vae = VAELoader.load_vae("ae.safetensors")[0]
        
        clip_device_str = "cuda:1" if DUAL_GPU else "default"
        clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2", device=clip_device_str)[0]
        
        if DUAL_GPU:
            clip.patcher.model.to(device_1)
            clip.patcher.load_device = device_1
            clip.patcher.current_device = device_1
            clip.patcher.offload_device = device_1 
            
    _Z_IMAGE_MODELS = (unet, clip, vae)
    return _Z_IMAGE_MODELS

if not (LOW_GPU and not DUAL_GPU):
    load_z_models()


# ======================================================================
# HELPER FUNCTIONS 
# ======================================================================
SUPPORTED_ASPECTS = [
    (1024, 1024), (1152, 896), (896, 1152), (1152, 864), (864, 1152), 
    (1248, 832), (832, 1248), (1280, 720), (720, 1280), (1344, 576), (576, 1344)
]

def pil_to_tensor(img):
    img = img.convert("RGB")
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

def pil_to_mask(mask):
    mask = mask.convert("L")
    return torch.from_numpy(np.array(mask).astype(np.float32) / 255.0).unsqueeze(0)

def ensure_mod_8(img, mask):
    w, h = img.size
    new_w, new_h = w - (w % 8), h - (h % 8)
    if (w, h) != (new_w, new_h):
        img = img.crop((0, 0, new_w, new_h))
        mask = mask.crop((0, 0, new_w, new_h))
    return img, mask

def fit_to_closest_aspect_ratio(img, mask):
    w, h = img.size
    best_res = min(SUPPORTED_ASPECTS, key=lambda r: abs((r[0]/r[1]) - (w/h)))
    target_w, target_h = best_res
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    mask = mask.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w - target_w) // 2, (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    mask = mask.crop((left, top, left + target_w, top + target_h))
    return img, mask

# ======================================================================
# INPAINT GENERATION FUNCTION (PHASE-SEQUENTIAL LOADING)
# ======================================================================
@torch.inference_mode()
def edit_image(init_image_path, mask_image_path, pos_prompt, neg_prompt, steps, cfg, seed, denoise, unet_opt, clip_opt, vae_opt):
    print(f"\n[DEBUG] ⚙️ Generating edit... Seed: {seed}")
    
    raw_image = Image.open(init_image_path)
    mask_image = Image.open(mask_image_path)
    image_tensor = pil_to_tensor(raw_image)
    mask_tensor = pil_to_mask(mask_image)
    del raw_image, mask_image

    is_sequential = (unet_opt is None)

    # --- PHASE 1: TEXT ENCODING ---
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2", device="default")[0] if is_sequential else clip_opt
    positive = CLIPTextEncode.encode(clip, pos_prompt)[0]
    negative = CLIPTextEncode.encode(clip, neg_prompt)[0]
    if is_sequential:
        del clip
        aggressive_clean()

    # --- PHASE 2: LATENT CONDITIONING ---
    vae = VAELoader.load_vae("ae.safetensors")[0] if is_sequential else vae_opt
    pos_cond, neg_cond, latent_image = InpaintModelConditioning.encode(
        positive=positive, negative=negative, vae=vae, pixels=image_tensor, mask=mask_tensor, noise_mask=True  
    )
    del image_tensor, mask_tensor, positive, negative
    if is_sequential:
        del vae
        aggressive_clean()

    # --- PHASE 3: UNET SAMPLING ---
    unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0] if is_sequential else unet_opt
    patched_unet = ModelSamplingAuraFlow.patch_aura(model=unet, shift=3)[0]
    
    samples = KSampler.sample(
        model=patched_unet, seed=seed, steps=steps, cfg=cfg, 
        sampler_name="euler_ancestral", scheduler="simple", 
        positive=pos_cond, negative=neg_cond, latent_image=latent_image, denoise=denoise
    )[0]
    del pos_cond, neg_cond, latent_image
    
    if is_sequential:
        del unet, patched_unet
        aggressive_clean()

    # --- PHASE 4: VAE DECODING ---
    vae = VAELoader.load_vae("ae.safetensors")[0] if is_sequential else vae_opt
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    result_img = Image.fromarray(np.clip(decoded.cpu().numpy()[0] * 255, 0, 255).astype(np.uint8))
    
    if is_sequential:
        del vae, samples, decoded
        aggressive_clean()
    
    return result_img

# ======================================================================
# GRADIO BACKEND LOGIC
# ======================================================================
def process_gradio(editor_data, pos_prompt, neg_prompt, steps, cfg, seed, denoise, mask_expand, mask_blur):
    start_time = time.time()
    
    if editor_data is None or editor_data.get("background") is None:
        raise gr.Error("Please upload an image!")
    if len(editor_data.get("layers", [])) == 0:
        raise gr.Error("Please draw a mask on the image!")

    force_resize = True if (LOW_GPU and not DUAL_GPU) else False

    bg_img = Image.fromarray(editor_data["background"]).convert("RGB")
    
    # Extract Mask Data
    mask_alpha = np.array(editor_data["layers"][0])[:, :, 3]
    mask_img = Image.fromarray((mask_alpha > 0).astype(np.uint8) * 255).convert("L")
    
    if mask_img.size != bg_img.size: 
        mask_img = mask_img.resize(bg_img.size, Image.LANCZOS)

    # 1. Expand (Dilate) the mask to give the model room to stitch edges
    if mask_expand > 0:
        for _ in range(int(mask_expand)):
            mask_img = mask_img.filter(ImageFilter.MaxFilter(3)) # Expands white boundary by 1px per iteration
            
    # 2. Feather (Blur) the edges to create a soft, seamless light transition
    if mask_blur > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=mask_blur))

    # Apply Resize bounds
    if force_resize:
        bg_img, mask_img = fit_to_closest_aspect_ratio(bg_img, mask_img)
    else:
        bg_img, mask_img = ensure_mod_8(bg_img, mask_img)

    safe_prompt = re.sub(r'[^a-zA-Z0-9_-]', '_', pos_prompt)[:25] or "image"
    uid = uuid.uuid4().hex[:6]
    
    result_dir = f"{root_path}/results"
    os.makedirs(result_dir, exist_ok=True)
    
    orig_path = f"{result_dir}/{safe_prompt}_{uid}_original.png"
    mask_path = f"{result_dir}/{safe_prompt}_{uid}_mask.png"
    out_path = f"{result_dir}/{safe_prompt}_{uid}_output.png"

    bg_img.save(orig_path)
    mask_img.save(mask_path)
    
    del bg_img, mask_img, mask_alpha, editor_data
    aggressive_clean()
    
    current_seed = random.randint(1, 1000000) if seed == 0 else int(seed)
    
    loaded_unet, loaded_clip, loaded_vae = load_z_models()
    output_img = edit_image(orig_path, mask_path, pos_prompt, neg_prompt, int(steps), float(cfg), current_seed, float(denoise), loaded_unet, loaded_clip, loaded_vae)
    output_img.save(out_path)
    
    duration = round(time.time() - start_time, 2)
    return output_img, out_path, str(current_seed), f"{duration} seconds"


# ======================================================================
# GRADIO USER INTERFACE
# ======================================================================
custom_css = """.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"""
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎨 Z-Image-Turbo Image Inpainting</h1>
        </div>""")
    
    with gr.Row():
        with gr.Column(scale=1):
            editor = gr.ImageEditor(
                label="Upload Image and Draw Mask", brush=gr.Brush(colors=["#ffffff"], default_size=30), type="numpy", height=500
            )
            # Updated Placeholder to help with context
            pos_prompt = gr.Textbox(label="Positive Prompt", placeholder="Describe the WHOLE image, including the edit (e.g., 'A happy girl smiling, wearing a beautiful white bride dress, blurred city background')", lines=2)
            generate_btn = gr.Button("✨ Generate", variant="primary")
            
            with gr.Accordion("Image Settings", open=False):
                neg_prompt = gr.Textbox(
                    label="Negative Prompt", value="anime, bad hands, pixelated, drawing, CGI, AI, stable diffusion, action figure, 2d, cartoon, sketch, render, 3d, painting, digital art, fan art, 2d art, wax, doll-like, perfect skin, 2.5d, smooth skin, jewelry, hyperrealistic, ultrarealistic, ugly, extra fingers, extra limbs, extra hands, extra body, 6 toes, six toes, 6 fingers, six fingers, supernumerary fingers, watermark, logo, writing", lines=4
                )
                with gr.Row():
                    steps = gr.Slider(minimum=4, maximum=25, step=1, value=10, label="Steps")
                    cfg = gr.Slider(minimum=0.5, maximum=4.0, step=0.1, value=1.0, label="CFG Scale")
                
                with gr.Row():
                    mask_expand = gr.Slider(minimum=0, maximum=30, step=1, value=10, label="Mask Expansion (Fixes Seams)")
                    mask_blur = gr.Slider(minimum=0, maximum=30, step=1, value=12, label="Mask Feathering (Fixes Hard Edges)")
                
                with gr.Row():
                    denoise = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=1.0, label="Denoise Strength")
                    seed = gr.Number(value=0, label="Seed (0 for Random)", precision=0)

        with gr.Column(scale=1):
            output_image = gr.Image(label="Output Image", height=500, type="pil")
            with gr.Accordion("Nerd Info", open=True): 
                file_download = gr.File(label="Download Generated Image")
                used_seed = gr.Textbox(label="Seed Used", interactive=False)
                gen_duration = gr.Textbox(label="Generation Time", interactive=False)

    generate_btn.click(
        fn=process_gradio, inputs=[editor, pos_prompt, neg_prompt, steps, cfg, seed, denoise, mask_expand, mask_blur], outputs=[output_image, file_download, used_seed, gen_duration]
    )

demo.launch(debug=True, share=True, allowed_paths=[f"{root_path}/results"])
