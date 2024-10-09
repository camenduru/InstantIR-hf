import gradio as gr
import numpy as np
import random
#import spaces #[uncomment to use ZeroGPU]
from diffusers import DiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_repo_id = "stabilityai/sdxl-turbo" #Replace to the model you would like to use

if torch.cuda.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

pipe = DiffusionPipeline.from_pretrained(model_repo_id, torch_dtype=torch_dtype)
pipe = pipe.to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

#@spaces.GPU #[uncomment to use ZeroGPU]
def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, progress=gr.Progress(track_tqdm=True)):

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    generator = torch.Generator().manual_seed(seed)
    
    image = pipe(
        prompt = prompt, 
        negative_prompt = negative_prompt,
        guidance_scale = guidance_scale, 
        num_inference_steps = num_inference_steps, 
        width = width, 
        height = height,
        generator = generator
    ).images[0] 
    
    return image, seed

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
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
        # Text-to-Image Gradio Template
        """)
        
        with gr.Row():
            
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            
            run_button = gr.Button("Run", scale=0)
        
        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):
            
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                visible=False,
            )
            
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024, #Replace with defaults that work for your model
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024, #Replace with defaults that work for your model
                )
            
            with gr.Row():
                
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0, #Replace with defaults that work for your model
                )
                
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=2, #Replace with defaults that work for your model
                )
        
        gr.Examples(
            examples = examples,
            inputs = [prompt]
        )
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs = [result, seed]
    )

demo.queue().launch()