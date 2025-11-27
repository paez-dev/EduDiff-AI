# EduDiff — Generador Educativo con IA
import gradio as gr
import requests
import io
from PIL import Image

API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-dev"

def generate(prompt):
    """Genera imagen educativa."""
    if not prompt or not prompt.strip():
        return None
    
    full_prompt = f"{prompt}, educational infographic, clean design, professional, high quality"
    
    try:
        response = requests.post(
            API_URL,
            json={"inputs": full_prompt},
            timeout=120
        )
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            return image
        else:
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Descripción", placeholder="Ej: Célula vegetal con núcleo y cloroplastos"),
    outputs=gr.Image(label="Resultado"),
    title="🎓 EduDiff - Generador Educativo",
    description="Genera imágenes educativas con IA (FLUX). Escribe una descripción del contenido.",
    cache_examples=False,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
