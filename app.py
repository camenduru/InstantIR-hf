import os
import sys
print(os.path.dirname(__file__))
print(os.listdir(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import torch
import spaces

import numpy as np
import gradio as gr
from PIL import Image

from diffusers import DDPMScheduler
from schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler

from module.ip_adapter.utils import load_adapter_to_pipe
from pipelines.sdxl_instantir import InstantIRPipeline

from huggingface_hub import hf_hub_download


def resize_img(input_image, max_side=1024, min_side=768, width=None, height=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    # Prepare output size
    if width > 0 and height > 0:
        out_w, out_h = width, height
    elif width > 0:
        out_w = width
        out_h = round(h * width / w)
    elif height > 0:
        out_h = height
        out_w = round(w * height / h)
    else:
        out_w, out_h = w, h

    # Resize input to runtime size
    w, h = out_w, out_h
    if min(w, h) < min_side:
        ratio = min_side / min(w, h)
        w, h = round(ratio * w), round(ratio * h)
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        w, h = round(ratio * w), round(ratio * h)
    # Resize to cope with UNet and VAE operations
    w_resize_new = (w // base_pixel_number) * base_pixel_number
    h_resize_new = (h // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image, (out_w, out_h)


if not os.path.exists("models/adapter.pt"):
    hf_hub_download(repo_id="InstantX/InstantIR", filename="models/adapter.pt", local_dir=".")
if not os.path.exists("models/aggregator.pt"):
    hf_hub_download(repo_id="InstantX/InstantIR", filename="models/aggregator.pt", local_dir=".")
if not os.path.exists("models/previewer_lora_weights.bin"):
    hf_hub_download(repo_id="InstantX/InstantIR", filename="models/previewer_lora_weights.bin", local_dir=".")

device = "cuda" if torch.cuda.is_available() else "cpu"
sdxl_repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
dinov2_repo_id = "facebook/dinov2-large"
lcm_repo_id = "latent-consistency/lcm-lora-sdxl"

torch_dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

# Load pretrained models.
print("Initializing pipeline...")
pipe = InstantIRPipeline.from_pretrained(
    sdxl_repo_id,
    torch_dtype=torch_dtype,
)

# Image prompt projector.
print("Loading LQ-Adapter...")
load_adapter_to_pipe(
    pipe,
    "models/adapter.pt",
    dinov2_repo_id,
)

# Prepare previewer
lora_alpha = pipe.prepare_previewers("models")
print(f"use lora alpha {lora_alpha}")
lora_alpha = pipe.prepare_previewers(lcm_repo_id, use_lcm=True)
print(f"use lora alpha {lora_alpha}")
pipe.to(device=device, dtype=torch_dtype)
pipe.scheduler = DDPMScheduler.from_pretrained(sdxl_repo_id, subfolder="scheduler")
lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)

pipe.scheduler = DDPMScheduler.from_pretrained(
    sdxl_repo_id,
    subfolder="scheduler"
)
lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)
# Load weights.
print("Loading checkpoint...")
aggregator_state_dict = torch.load(
    "models/aggregator.pt",
    map_location="cpu"
)
pipe.aggregator.load_state_dict(aggregator_state_dict)
pipe.aggregator.to(device=device, dtype=torch_dtype)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1280
MIN_IMAGE_SIZE = 1024

PROMPT = "Photorealistic, highly detailed, hyper detailed photo - realistic maximum detail, 32k, \
ultra HD, extreme meticulous detailing, skin pore detailing, \
hyper sharpness, perfect without deformations, \
taken using a Canon EOS R camera, Cinematic, High Contrast, Color Grading. "

NEG_PROMPT = "blurry, out of focus, unclear, depth of field, over-smooth, \
sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, \
dirty, messy, worst quality, low quality, frames, painting, illustration, drawing, art, \
watermark, signature, jpeg artifacts, deformed, lowres"

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def unpack_pipe_out(preview_row, index):
    return preview_row[index][0]

def dynamic_preview_slider(sampling_steps):
    return gr.Slider(label="Restoration Previews", value=sampling_steps-1, minimum=0, maximum=sampling_steps-1, step=1)

def dynamic_guidance_slider(sampling_steps):
    return gr.Slider(label="Start Free Rendering", value=sampling_steps, minimum=0, maximum=sampling_steps, step=1)

def show_final_preview(preview_row):
    return preview_row[-1][0]

@spaces.GPU(duration=70)
def instantir_restore(
    lq, prompt="", steps=30, cfg_scale=7.0, guidance_end=1.0,
    creative_restoration=False, seed=3407, height=None, width=None, preview_start=0.0):
    if creative_restoration:
        if "lcm" not in pipe.unet.active_adapters():
            pipe.unet.set_adapter('lcm')
    else:
        if "previewer" not in pipe.unet.active_adapters():
            pipe.unet.set_adapter('previewer')

    if isinstance(guidance_end, int):
        guidance_end = guidance_end / steps
    elif guidance_end > 1.0:
        guidance_end = guidance_end / steps
    if isinstance(preview_start, int):
        preview_start = preview_start / steps
    elif preview_start > 1.0:
        preview_start = preview_start / steps

    lq, out_size = resize_img(lq, width=width, height=height)
    lq = [lq]
    generator = torch.Generator(device=device).manual_seed(seed)
    timesteps = [
        i * (1000//steps) + pipe.scheduler.config.steps_offset for i in range(0, steps)
    ]
    timesteps = timesteps[::-1]

    prompt = PROMPT if len(prompt)==0 else prompt
    neg_prompt = NEG_PROMPT

    out = pipe(
        prompt=[prompt]*len(lq),
        image=lq,
        num_inference_steps=steps,
        generator=generator,
        timesteps=timesteps,
        negative_prompt=[neg_prompt]*len(lq),
        guidance_scale=cfg_scale,
        control_guidance_end=guidance_end,
        preview_start=preview_start,
        previewer_scheduler=lcm_scheduler,
        return_dict=False,
        save_preview_row=True,
    )
    out[0][0] = out[0][0].resize([out_size[0], out_size[1]], Image.BILINEAR)
    for i, preview_tuple in enumerate(out[1]):
        preview_tuple[0] = preview_tuple[0].resize([out_size[0], out_size[1]], Image.BILINEAR)
        preview_tuple.append(f"preview_{i}")
    return out[0][0], out[1]

css="""
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # InstantIR: Blind Image Restoration with Instant Generative Reference.
    ### **Official 🤗 Gradio demo of [InstantIR](https://github.com/instantX-research/InstantIR).**
    ### **InstantIR can not only help you restore your broken image, but also capable of imaginative re-creation following your text prompts. See advance usage for more details!**
    ## Basic usage: revitalize your image
    1. Upload an image you want to restore;
    2. By default InstantIR will restore your image at original size, you can change output size by setting `Height` and `Width` according to your requirements;
    3. Optionally, tune the `Steps` `CFG Scale` parameters. Typically higher steps lead to better results, but less than 50 is recommended for efficiency;
    4. Click `InstantIR magic!`.
    """)
    with gr.Row():
        with gr.Column():
            lq_img = gr.Image(label="Low-quality image", type="pil")
            with gr.Row():
                restore_btn = gr.Button("InstantIR magic!")
                clear_btn = gr.ClearButton()
            with gr.Row():
                steps = gr.Number(label="Steps", value=30, step=1)
                cfg_scale = gr.Number(label="CFG Scale", value=7.0, step=0.1)
            with gr.Row():
                height = gr.Number(label="Height", step=1, maximum=MAX_IMAGE_SIZE)
                width = gr.Number(label="Width", step=1, maximum=MAX_IMAGE_SIZE)
                seed = gr.Number(label="Seed", value=42, step=1)
            guidance_end = gr.Slider(label="Start Free Rendering", value=30, minimum=0, maximum=30, step=1)
            preview_start = gr.Slider(label="Preview Start", value=0, minimum=0, maximum=30, step=1)
            mode = gr.Checkbox(label="Creative Restoration", value=False)
            prompt = gr.Textbox(label="Restoration prompts (Optional)", placeholder="")
            gr.Examples(
                examples = [
                    "./examples/wukong.png", "./examples/lady.png", "./examples/man.png", "./examples/dog.png", "./examples/panda.png", "./examples/sculpture.png", "./examples/cottage.png", "./examples/Naruto.png", "./examples/Konan.png"
                    ],
                inputs = [lq_img]
            )
        with gr.Column():
            output = gr.Image(label="InstantIR restored", type="pil")
            index = gr.Slider(label="Restoration Previews", value=29, minimum=0, maximum=29, step=1)
            preview = gr.Image(label="Preview", type="pil")

    pipe_out = gr.Gallery(visible=False)
    clear_btn.add([lq_img, output, preview])
    restore_btn.click(
        instantir_restore, inputs=[
            lq_img, prompt, steps, cfg_scale, guidance_end,
            mode, seed, height, width, preview_start,
        ],
        outputs=[output, pipe_out], api_name="InstantIR"
    )
    steps.change(dynamic_guidance_slider, inputs=steps, outputs=guidance_end)
    output.change(dynamic_preview_slider, inputs=steps, outputs=index)
    index.release(unpack_pipe_out, inputs=[pipe_out, index], outputs=preview)
    output.change(show_final_preview, inputs=pipe_out, outputs=preview)
    gr.Markdown(
    """
    ## Advance usage:
    ### Browse restoration variants:
    1. After InstantIR processing, drag the `Restoration Previews` slider to explore other in-progress versions;
    2. If you like one of them, set the `Start Free Rendering` slider to the same value to get a more refined result.
    ### Creative restoration:
    1. Check the `Creative Restoration` checkbox;
    2. Input your text prompts in the `Restoration prompts` textbox;
    3. Set `Start Free Rendering` slider to a medium value (around half of the `steps`) to provide adequate room for InstantIR creation.
    """)
    gr.Markdown(
    """
    ## Citation
    If InstantIR is helpful to your work, please cite our paper via:
    ```
    @article{huang2024instantir,
        title={InstantIR: Blind Image Restoration with Instant Generative Reference},
        author={Huang, Jen-Yuan and Wang, Haofan and Wang, Qixun and Bai, Xu and Ai, Hao and Xing, Peng and Huang, Jen-Tse},
        journal={arXiv preprint arXiv:2410.06551},
        year={2024}
    }
    ```
    """)

demo.queue().launch()