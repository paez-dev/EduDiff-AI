# ═══════════════════════════════════════════════════════════════════════════════
# EduDiff — Generador de Material Educativo con IA Generativa
# Versión para Hugging Face Spaces (usa Inference API)
# ═══════════════════════════════════════════════════════════════════════════════

import gradio as gr
from huggingface_hub import InferenceClient
from PIL import Image
import io
import os

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

# Cliente de inferencia (usa la API gratuita de HF)
client = InferenceClient()

# Modelo a usar
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Estilos educativos
STYLES = {
    "📊 Infografía": "professional infographic, clean design, labeled diagram, white background, educational, high quality",
    "🎨 Ilustración": "digital illustration, vibrant colors, educational style, child-friendly, clear",
    "🔬 Científico": "scientific illustration, detailed anatomy, textbook quality, labeled parts, precise",
    "📐 Diagrama": "technical diagram, clean lines, flowchart style, organized layout, schematic"
}

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE GENERACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def generate_image(prompt, style, guidance):
    """Genera una imagen educativa usando la API de HuggingFace."""
    
    if not prompt or not prompt.strip():
        return None, "⚠️ Por favor ingresa una descripción"
    
    # Construir prompt completo
    style_suffix = STYLES.get(style, STYLES["📊 Infografía"])
    full_prompt = f"{prompt}, {style_suffix}"
    negative = "blurry, bad quality, watermark, text errors, ugly, deformed"
    
    try:
        # Llamar a la API de inferencia
        image = client.text_to_image(
            prompt=full_prompt,
            negative_prompt=negative,
            model=MODEL_ID,
            guidance_scale=float(guidance),
            num_inference_steps=25
        )
        
        return image, "✅ Imagen generada correctamente"
        
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return None, "⚠️ Límite de API alcanzado. Espera unos segundos e intenta de nuevo."
        elif "loading" in error_msg.lower():
            return None, "⏳ El modelo se está cargando. Espera 30 segundos e intenta de nuevo."
        else:
            return None, f"❌ Error: {error_msg[:100]}"

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ GRADIO
# ═══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="EduDiff - Generador Educativo",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # 🎓 EduDiff — Generador de Material Educativo
    
    Crea imágenes educativas usando **Stable Diffusion XL**.
    
    **Instrucciones:** Describe el contenido que necesitas y selecciona un estilo.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_box = gr.Textbox(
                label="📝 Descripción",
                placeholder="Ej: Diagrama de célula vegetal con cloroplastos, núcleo y pared celular etiquetados",
                lines=3
            )
            
            style_radio = gr.Radio(
                choices=list(STYLES.keys()),
                value="📊 Infografía",
                label="🎨 Estilo"
            )
            
            guidance_slider = gr.Slider(
                minimum=5, maximum=12, value=7.5, step=0.5,
                label="Guidance Scale"
            )
            
            gen_btn = gr.Button("🚀 Generar Imagen", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_img = gr.Image(label="Resultado", type="pil", height=400)
            status_txt = gr.Textbox(label="Estado", interactive=False)
    
    # Ejemplos
    gr.Examples(
        examples=[
            ["Diagrama de célula animal mostrando núcleo, mitocondrias y membrana celular", "🔬 Científico", 7.5],
            ["Ciclo del agua con evaporación, condensación y precipitación", "📊 Infografía", 7.5],
            ["Sistema solar con los planetas en orden", "🎨 Ilustración", 7.5],
            ["Pirámide alimenticia con frutas, verduras y proteínas", "📐 Diagrama", 7.5],
        ],
        inputs=[prompt_box, style_radio, guidance_slider],
        label="💡 Ejemplos"
    )
    
    # Evento
    gen_btn.click(
        fn=generate_image,
        inputs=[prompt_box, style_radio, guidance_slider],
        outputs=[output_img, status_txt]
    )
    
    gr.Markdown("""
    ---
    ### ℹ️ Información
    
    - **Modelo:** Stable Diffusion XL (via Hugging Face Inference API)
    - **Uso:** Material didáctico, infografías, diagramas educativos
    - **Nota:** El contenido debe verificarse antes de uso educativo
    
    *Proyecto EA3 - Generación de Contenido con IA Generativa*
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# INICIO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch()
