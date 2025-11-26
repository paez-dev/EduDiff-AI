from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

MODEL_ID = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar pipeline optimizado
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    safety_checker=None,
    device_map="auto" if device=="cuda" else None,
    low_cpu_mem_usage=True
)

def generar(prompt, estilo, steps, scale):
    full_prompt = f"{prompt}, estilo: {estilo}, clean vector illustration, educational infographic, clear labels"
    
    # Autocast solo en CUDA
    if device == "cuda":
        with torch.autocast("cuda"):
            image = pipe(full_prompt, guidance_scale=float(scale), num_inference_steps=int(steps)).images[0]
    else:
        image = pipe(full_prompt, guidance_scale=float(scale), num_inference_steps=int(steps)).images[0]
    return image

iface = gr.Interface(
    fn=generar,
    inputs=[
        gr.Textbox(label="Prompt", lines=3, placeholder="Ej: Diagrama de célula vegetal con etiquetas..."),
        gr.Dropdown(
            ["Infografía", "Ilustración", "Dibujo escolar", "Realista suave"],
            value="Infografía",
            label="Estilo"
        ),
        gr.Slider(10, 50, value=15, step=1, label="Num Inference Steps"),
        gr.Slider(1, 15, value=7.5, step=0.1, label="Guidance Scale")
    ],
    outputs=gr.Image(type="pil"),
    title="EduDiff - Generador de Imágenes Educativas",
    description="""
Genera infografías, diagramas y esquemas educativos a partir de un prompt.  
Ajusta 'Steps' y 'Guidance Scale' para controlar calidad y creatividad.
""",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()