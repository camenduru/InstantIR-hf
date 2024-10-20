import os
print(os.listdir('.'))
import torch
import random
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

from diffusers import (
    DDPMScheduler,
    StableDiffusionXLPipeline
)
from schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from transformers import (
    AutoImageProcessor, AutoModel
)

from module.ip_adapter.utils import init_ip_adapter_in_unet
from module.ip_adapter.resampler import Resampler
from module.aggregator import Aggregator
from pipelines.sdxl_instantir import InstantIRPipeline, LCM_LORA_MODULES, PREVIEWER_LORA_MODULES


transform = transforms.Compose([
    transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(1024),
])

device = "cuda" if torch.cuda.is_available() else "cpu"
sdxl_repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
instantir_repo_id = "instantx/instantir"
dinov2_repo_id = "facebook/dinov2-large"

if torch.cuda.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

print("Loading vision encoder...")
image_encoder = AutoModel.from_pretrained(dinov2_repo_id, torch_dtype=torch_dtype)
image_processor = AutoImageProcessor.from_pretrained(dinov2_repo_id)

print("Loading SDXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    sdxl_repo_id,
    torch_dtype=torch.float16,
)
unet = pipe.unet

print("Initializing Aggregator...")
aggregator = Aggregator.from_unet(unet, load_weights_from_unet=False)

print("Loading LQ-Adapter...")
image_proj_model = Resampler(
    dim=1280,
    depth=4,
    dim_head=64,
    heads=20,
    num_queries=64,
    embedding_dim=image_encoder.config.hidden_size,
    output_dim=unet.config.cross_attention_dim,
    ff_mult=4
)
init_ip_adapter_in_unet(
    unet,
    image_proj_model,
    "InstantX/InstantIR/adapter.pt",
    adapter_tokens=64,
)
print("Initializing InstantIR...")
pipe = InstantIRPipeline(
        pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2,
        unet, aggregator, pipe.scheduler, feature_extractor=image_processor, image_encoder=image_encoder,
)

# Add Previewer LoRA.
lora_state_dict, alpha_dict = StableDiffusionXLPipeline.lora_state_dict(
    "InstantX/InstantIR/previewer_lora_weights.bin",
    # weight_name="previewer_lora_weights.bin",

)
unet_state_dict = {
    f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")
}
unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
lora_state_dict = dict()
for k, v in unet_state_dict.items():
    if "ip" in k:
        k = k.replace("attn2", "attn2.processor")
        lora_state_dict[k] = v
    else:
        lora_state_dict[k] = v
if alpha_dict:
    lora_alpha = next(iter(alpha_dict.values()))
else:
    lora_alpha = 1
print(f"use lora alpha {lora_alpha}")
lora_config = LoraConfig(
    r=64,
    target_modules=PREVIEWER_LORA_MODULES,
    lora_alpha=lora_alpha,
    lora_dropout=0.0,
)

# Add LCM LoRA.
lora_state_dict, alpha_dict = StableDiffusionXLPipeline.lora_state_dict(
    "latent-consistency/lcm-lora-sdxl"
)
unet_state_dict = {
    f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")
}
unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
if alpha_dict:
    lora_alpha = next(iter(alpha_dict.values()))
else:
    lora_alpha = 1
print(f"use lora alpha {lora_alpha}")
lora_config = LoraConfig(
    r=64,
    target_modules=LCM_LORA_MODULES,
    lora_alpha=lora_alpha,
    lora_dropout=0.0,
)

unet.add_adapter(lora_config, "lcm")
incompatible_keys = set_peft_model_state_dict(unet, unet_state_dict, adapter_name="lcm")
if incompatible_keys is not None:
    # check only for unexpected keys
    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
    missing_keys = getattr(incompatible_keys, "missing_keys", None)
    if unexpected_keys:
        raise ValueError(
            f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
            f" {unexpected_keys}. "
        )

unet.disable_adapters()
pipe.scheduler = DDPMScheduler.from_pretrained(
    sdxl_repo_id,
    subfolder="scheduler"
)
lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)
# Load weights.
print("Loading checkpoint...")
aggregator_state_dict = torch.load(
    "InstantX/InstantIR/aggregator.pt",
    map_location="cpu"
)
aggregator.load_state_dict(aggregator_state_dict, strict=True)
aggregator.to(dtype=torch.float16)
unet.to(dtype=torch.float16)
pipe=pipe.to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def unpack_pipe_out(preview_row, index):
    return preview_row[index][0]

def dynamic_preview_slider(sampling_steps):
    print(sampling_steps)
    return gr.Slider(label="Restoration Previews", value=sampling_steps-1, minimum=0, maximum=sampling_steps-1, step=1)

def dynamic_guidance_slider(sampling_steps):
    return gr.Slider(label="Start Free Rendering", value=sampling_steps, minimum=0, maximum=sampling_steps, step=1)

def show_final_preview(preview_row):
    return preview_row[-1][0]

# @spaces.GPU #[uncomment to use ZeroGPU]
def instantir_restore(lq, prompt="", steps=30, cfg_scale=7.0, guidance_end=1.0, creative_restoration=False, seed=3407):
    if creative_restoration:
        if "lcm" not in pipe.unet.active_adapters():
            pipe.unet.set_adapter('lcm')
    else:
        if "previewer" not in pipe.unet.active_adapters():
            pipe.unet.set_adapter('previewer')

    if isinstance(guidance_end, int):
        guidance_end = guidance_end / steps
    with torch.no_grad(): lq = [transform(lq)]
    generator = torch.Generator(device=device).manual_seed(seed)

    out = pipe(
        prompt=[prompt]*len(lq),
        image=lq,
        ip_adapter_image=[lq],
        num_inference_steps=steps,
        generator=generator,
        controlnet_conditioning_scale=1.0,
        # negative_original_size=(256,256),
        # negative_target_size=(1024,1024),
        negative_prompt=[""]*len(lq),
        guidance_scale=cfg_scale,
        control_guidance_end=guidance_end,
        # control_guidance_start=0.5,
        previewer_scheduler=lcm_scheduler,
        return_dict=False,
        save_preview_row=True,
        # reference_latent = reference_latents,
        # output_type='pt'
    )
    for i, preview_img in enumerate(out[1]):
        preview_img.append(f"preview_{i}")
    return out[0][0], out[1]

examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
    """
    # InstantIR: Blind Image Restoration with Instant Generative Reference.

    ### **Official 🤗 Gradio demo of [InstantIR](https://arxiv.org/abs/2410.06551).**
    ### **InstantIR can not only help you restore your broken image, but also capable of imaginative re-creation following your text prompts. See advance usage for more details!**
    ## Basic usage: revitalize your image
    1. Upload an image you want to restore;
    2. Optionally, tune the `Steps` `CFG Scale` parameters. Typically higher steps lead to better results, but less than 50 is recommended for efficiency;
    3. Click `InstantIR magic!`.
    """)
    with gr.Row():
        lq_img = gr.Image(label="Low-quality image", type="pil")
        with gr.Column(elem_id="col-container"):
            with gr.Row():
                steps = gr.Number(label="Steps", value=20, step=1)
                cfg_scale = gr.Number(label="CFG Scale", value=7.0, step=0.1)
                seed = gr.Number(label="Seed", value=42, step=1)
            # guidance_start = gr.Slider(label="Guidance Start", value=1.0, minimum=0.0, maximum=1.0, step=0.05)
            guidance_end = gr.Slider(label="Start Free Rendering", value=20, minimum=0, maximum=20, step=1)
            prompt = gr.Textbox(
                label="Restoration prompts (Optional)", show_label=False,
                placeholder="Restoration prompts (Optional)", value='',
                # container=False,
            )
            mode = gr.Checkbox(label="Creative Restoration", value=False)
    # with gr.Accordion("Advanced Settings", open=False):
    with gr.Row():
        with gr.Row():
            restore_btn = gr.Button("InstantIR magic!")
            clear_btn = gr.ClearButton()
        index = gr.Slider(label="Restoration Previews", value=19, minimum=0, maximum=19, step=1)
    with gr.Row():
        output = gr.Image(label="InstantIR restored", type="pil")
        preview = gr.Image(label="Preview", type="pil")
    # gr.Examples(
    #     examples = examples,
    #     inputs = [prompt]
    # )
    # gr.on(
    #     triggers=[restore_btn.click, prompt.submit],
    #     fn = infer,
    #     inputs = [prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
    #     outputs = [result, seed]
    # )
    pipe_out = gr.Gallery(visible=False)
    clear_btn.add([lq_img, output, preview])
    restore_btn.click(instantir_restore, inputs=[lq_img, prompt, steps, cfg_scale, guidance_end, mode, seed], outputs=[output, pipe_out], api_name="InstantIR")
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
    
    ## Examples
    Here are some examplar usage of InstantIR:
    """)
    # examples = gr.Gallery(label="Examples")

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

demo.queue().launch(debug=True)