from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

# Configuración del modelo
MODEL_ID = "runwayml/stable-diffusion-v1-5"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar el pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
pipe = pipe.to(device)
pipe.safety_checker = None  # Opcional, desactivar si quieres evitar el safety checker

# Función de generación
def generar(prompt, estilo, steps, scale):
    full_prompt = f"{prompt}, estilo: {estilo}, clean vector illustration, educational infographic, clear labels"
    with torch.autocast(device):
        image = pipe(full_prompt, guidance_scale=float(scale), num_inference_steps=int(steps)).images[0]
    return image

# Interfaz Gradio
title = "EduDiff - Generador de Imágenes Educativas"
description = """
Genera infografías, diagramas y esquemas educativos a partir de un prompt.  
Ajusta 'Steps' y 'Guidance Scale' para controlar calidad y creatividad.
"""

iface = gr.Interface(
    fn=generar,
    inputs=[
        gr.Textbox(label="Prompt", lines=3, placeholder="Ej: Diagrama de célula vegetal con etiquetas..."),
        gr.Dropdown(
            ["Infografía", "Ilustración", "Dibujo escolar", "Realista suave"],
            value="Infografía",
            label="Estilo"
        ),
        gr.Slider(10, 50, value=30, step=1, label="Num Inference Steps"),
        gr.Slider(1, 15, value=7.5, step=0.1, label="Guidance Scale")
    ],
    outputs=gr.Image(type="pil"),
    title=title,
    description=description,
    allow_flagging="never"
)

# Lanzamiento de la app
if __name__ == "__main__":
    iface.launch()