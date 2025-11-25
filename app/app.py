from diffusers import StableDiffusionPipeline
import torch
import gradio as gr
import os

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = os.environ.get("LORA_PATH", "")  # dejar vacío si no aplicas LoRA

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
# Si se usó LoRA con atención de weights (ejemplo), cargarlos aquí (ajusta según método)
if LORA_PATH:
    try:
        pipe.unet.load_attn_procs(LORA_PATH)
        print("LoRA cargado:", LORA_PATH)
    except Exception as e:
        print("No se pudo cargar LoRA:", e)

pipe = pipe.to(device)
pipe.safety_checker = None  # opcional: según políticas del espacio

def generate(prompt, style, steps, scale):
    full_prompt = f"{prompt}, estilo: {style}, clean vector illustration, educational infographic, clear labels"
    image = pipe(full_prompt, guidance_scale=float(scale), num_inference_steps=int(steps)).images[0]
    return image

title = "EduDiff - Generador de Imágenes Educativas"
description = "Genera infografías y diagramas educativos a partir de prompts. Ajusta 'steps' y 'guidance scale' para controlar calidad y creatividad."

iface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", lines=3, placeholder="Ej: Diagrama de célula vegetal con etiquetas..."),
        gr.Dropdown(["Infografía", "Ilustración", "Dibujo escolar", "Realista suave"], value="Infografía", label="Estilo"),
        gr.Slider(10, 50, value=30, label="Num Inference Steps"),
        gr.Slider(1, 15, value=7.5, label="Guidance Scale"),
    ],
    outputs=gr.Image(type="pil"),
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)