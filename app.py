# EduDiff — Generador de Material Educativo con IA
# Versión mínima para Hugging Face Spaces

import gradio as gr
from huggingface_hub import InferenceClient
from PIL import Image

# Cliente de inferencia
client = InferenceClient()

def generate(prompt, style):
    """Genera imagen educativa."""
    if not prompt:
        return None
    
    styles = {
        "Infografía": "infographic, clean design, educational, white background",
        "Ilustración": "illustration, colorful, child-friendly, educational",
        "Científico": "scientific diagram, detailed, textbook style, labeled",
        "Diagrama": "diagram, flowchart, organized, schematic"
    }
    
    full_prompt = f"{prompt}, {styles.get(style, styles['Infografía'])}, high quality"
    
    try:
        image = client.text_to_image(
            prompt=full_prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )
        return image
    except Exception as e:
        print(f"Error: {e}")
        return None

# Interfaz simple
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Descripción", placeholder="Ej: Célula vegetal con núcleo y cloroplastos", lines=2),
        gr.Dropdown(["Infografía", "Ilustración", "Científico", "Diagrama"], value="Infografía", label="Estilo")
    ],
    outputs=gr.Image(label="Resultado"),
    title="🎓 EduDiff - Generador Educativo",
    description="Genera imágenes educativas con IA. Escribe una descripción y selecciona un estilo.",
    examples=[
        ["Diagrama de célula animal con núcleo y mitocondrias", "Científico"],
        ["Ciclo del agua con evaporación y precipitación", "Infografía"],
        ["Sistema solar con planetas", "Ilustración"],
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
