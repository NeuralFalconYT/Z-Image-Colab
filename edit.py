#@title Run Gradio app Z-Image-Turbo Image Inpaint 
# root_path="/content"
# %cd $root_path/ComfyUI

import os,sys,gc,random,time,uuid,re
import torch
import numpy as np
from PIL import Image,ImageFilter
import gradio as gr
import psutil
import importlib
import importlib.util
import shutil
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:128"

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32=True



comfyui_path=f"{root_path}/ComfyUI"

if comfyui_path in sys.path:
    sys.path.remove(comfyui_path)

sys.path.insert(0,comfyui_path)

for k in list(sys.modules.keys()):
    if k=="utils" or k.startswith("utils."):
        del sys.modules[k]

importlib.invalidate_caches()

utils_path=os.path.join(comfyui_path,"utils","__init__.py")

spec=importlib.util.spec_from_file_location(
    "utils",
    utils_path,
    submodule_search_locations=[os.path.join(comfyui_path,"utils")]
)

utils_module=importlib.util.module_from_spec(spec)
sys.modules["utils"]=utils_module
spec.loader.exec_module(utils_module)



import comfy.model_management
import nest_asyncio
nest_asyncio.apply()

import asyncio
import server
import execution
import nodes

loop=asyncio.get_event_loop()
server_instance=server.PromptServer(loop)
execution.PromptQueue(server_instance)

res=nodes.init_extra_nodes()

if asyncio.iscoroutine(res):
    loop.run_until_complete(res)

from nodes import NODE_CLASS_MAPPINGS


# ---------------------------------------------------
# LOW VRAM MODE
# ---------------------------------------------------

comfy.model_management.set_vram_state(
    comfy.model_management.VRAMState.LOW_VRAM
)

comfy.model_management.force_full_precision=False


# ---------------------------------------------------
# NODE CLASSES
# ---------------------------------------------------

UNETLoader=NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader=NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader=NODE_CLASS_MAPPINGS["VAELoader"]()

CLIPTextEncode=NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler=NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode=NODE_CLASS_MAPPINGS["VAEDecode"]()

InpaintModelConditioning=NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
ModelSamplingAuraFlow=NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()


# ---------------------------------------------------
# MEMORY DEBUG
# ---------------------------------------------------

def mem(tag=""):

    ram=psutil.virtual_memory().used/1024**3
    vram=torch.cuda.memory_allocated()/1024**3

    print(f"[MEM] {tag} | RAM {ram:.2f}GB | VRAM {vram:.2f}GB")


# ---------------------------------------------------
# MODEL LOAD (CPU FIRST)
# ---------------------------------------------------

MODELS=None

def load_models():

    global MODELS

    if MODELS is not None:
        return MODELS

    # print("Loading models...")

    with torch.inference_mode():

        unet=UNETLoader.load_unet(
            "z-image-turbo-fp8-e4m3fn.safetensors",
            "fp8_e4m3fn_fast"
        )[0]

        clip=CLIPLoader.load_clip(
            "qwen_3_4b.safetensors",
            type="lumina2",
            device="cpu"
        )[0]

        vae=VAELoader.load_vae("ae.safetensors")[0]

        clip.patcher.model.to("cpu")
        vae.first_stage_model.to("cpu")

    MODELS=(unet,clip,vae)

    # mem("models loaded")

    return MODELS


# ---------------------------------------------------
# CLEANUP
# ---------------------------------------------------

def clear():

    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # mem("cleanup")


# ---------------------------------------------------
# IMAGE UTILS
# ---------------------------------------------------

def pil_to_tensor(img):

    img=img.convert("RGB")

    return torch.from_numpy(
        np.array(img).astype(np.float32)/255
    ).unsqueeze(0)


def pil_to_mask(mask):

    mask=mask.convert("L")

    return torch.from_numpy(
        np.array(mask).astype(np.float32)/255
    ).unsqueeze(0)


# ---------------------------------------------------
# GENERATION
# ---------------------------------------------------

@torch.inference_mode()
def generate(img_path,mask_path,pos,neg,steps,cfg,seed,denoise):

    unet,clip,vae=load_models()

    # mem("start")

    image=Image.open(img_path)
    mask=Image.open(mask_path)

    image_t=pil_to_tensor(image)
    mask_t=pil_to_mask(mask)

    # CLIP encode on GPU

    clip.patcher.model.to("cuda")

    positive=CLIPTextEncode.encode(clip,pos)[0]
    negative=CLIPTextEncode.encode(clip,neg)[0]

    clip.patcher.model.to("cpu")

    clear()

    pos_c,neg_c,latent=InpaintModelConditioning.encode(
        positive=positive,
        negative=negative,
        vae=vae,
        pixels=image_t,
        mask=mask_t,
        noise_mask=True
    )

    # UNET GPU

    unet.model.to("cuda")

    clear()

    patched=ModelSamplingAuraFlow.patch_aura(
        model=unet,
        shift=3
    )[0]

    samples=KSampler.sample(
        model=patched,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name="euler_ancestral",
        scheduler="simple",
        positive=pos_c,
        negative=neg_c,
        latent_image=latent,
        denoise=denoise
    )[0]

    unet.model.to("cpu")

    clear()

    # VAE decode GPU

    vae.first_stage_model.to("cuda")

    decoded=VAEDecode.decode(vae,samples)[0]

    vae.first_stage_model.to("cpu")

    clear()

    img=Image.fromarray(
        np.clip(decoded.cpu().numpy()[0]*255,0,255).astype(np.uint8)
    )

    return img


# ---------------------------------------------------
# GRADIO
# ---------------------------------------------------

def process(editor,pos,neg,steps,cfg,seed,denoise):

    if editor is None:
        raise gr.Error("upload image")

    bg=Image.fromarray(editor["background"]).convert("RGB")

    alpha=np.array(editor["layers"][0])[:,:,3]

    mask=Image.fromarray((alpha>0).astype(np.uint8)*255)

    uid=uuid.uuid4().hex[:6]

    os.makedirs("results",exist_ok=True)

    ip=f"results/{uid}_img.png"
    mp=f"results/{uid}_mask.png"
    op=f"results/{uid}_out.png"

    bg.save(ip)
    mask.save(mp)

    seed=random.randint(1,999999) if seed==0 else int(seed)

    img=generate(ip,mp,pos,neg,steps,cfg,seed,denoise)

    img.save(op)
    drive_path="/content/gdrive/MyDrive/z_image_turbo_Inpainting/"
    if os.path.exists(drive_path):
      shutil.copy(op,drive_path)
    return img,op,str(seed)


with gr.Blocks() as demo:

    editor=gr.ImageEditor(type="numpy",height=500)

    pos=gr.Textbox(label="Prompt")

    neg=gr.Textbox(value="anime, cartoon")

    steps=gr.Slider(4,25,10)

    cfg=gr.Slider(0.5,4,1)

    denoise=gr.Slider(0.1,1,1)

    seed=gr.Number(value=0)

    btn=gr.Button("Generate")

    out=gr.Image()

    file=gr.File()

    seed_out=gr.Textbox()

    btn.click(
        process,
        [editor,pos,neg,steps,cfg,seed,denoise],
        [out,file,seed_out]
    )


demo.launch(share=True,debug=True)
