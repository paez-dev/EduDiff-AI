# ═══════════════════════════════════════════════════════════════════════════════
# EduDiff XL — Generador de Material Educativo con IA Generativa
# Versión simplificada para Hugging Face Spaces
# ═══════════════════════════════════════════════════════════════════════════════

import os
import gc
import torch
import gradio as gr
from PIL import Image

# Intentar importar diffusers (puede fallar en CPU limitada)
try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_ID = "runwayml/stable-diffusion-v1-5"
_device = "cuda" if torch.cuda.is_available() else "cpu"
_pipe = None

# Estilos educativos
STYLES = {
    "Infografía": "professional infographic, clean design, labeled diagram, white background, educational",
    "Ilustración": "digital illustration, vibrant colors, educational style, child-friendly",
    "Científico": "scientific illustration, detailed anatomy, textbook quality, labeled parts",
    "Diagrama": "technical diagram, clean lines, flowchart style, organized layout"
}

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Carga el modelo de Stable Diffusion."""
    global _pipe
    
    if _pipe is not None:
        return _pipe
    
    if not DIFFUSERS_AVAILABLE:
        return None
    
    try:
        dtype = torch.float16 if _device == "cuda" else torch.float32
        _pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None
        )
        _pipe.scheduler = DPMSolverMultistepScheduler.from_config(_pipe.scheduler.config)
        _pipe = _pipe.to(_device)
        return _pipe
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None


def generate_image(prompt, style, steps, guidance, seed):
    """Genera una imagen educativa."""
    
    if not prompt or not prompt.strip():
        return None, "⚠️ Por favor ingresa un prompt"
    
    # Cargar modelo
    pipe = load_model()
    
    if pipe is None:
        # Modo demo sin GPU - crear imagen placeholder
        img = Image.new('RGB', (512, 512), color=(240, 240, 250))
        return img, "⚠️ Modo demo: GPU no disponible. La imagen es un placeholder."
    
    # Construir prompt completo
    style_suffix = STYLES.get(style, STYLES["Infografía"])
    full_prompt = f"{prompt}, {style_suffix}"
    negative = "blurry, bad quality, text errors, watermark"
    
    # Configurar semilla
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device=_device).manual_seed(int(seed))
    
    try:
        # Generar
        with torch.inference_mode():
            result = pipe(
                prompt=full_prompt,
                negative_prompt=negative,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=generator,
                width=512,
                height=512
            )
        
        image = result.images[0]
        msg = f"✅ Generado con semilla: {seed}"
        return image, msg
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def clear_memory():
    """Libera memoria."""
    global _pipe
    if _pipe is not None:
        del _pipe
        _pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "✅ Memoria liberada"


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ GRADIO
# ═══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="EduDiff - Generador Educativo", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🎓 EduDiff — Generador de Material Educativo
    
    Genera imágenes educativas usando Inteligencia Artificial (Stable Diffusion).
    
    **Instrucciones:**
    1. Escribe una descripción del contenido educativo
    2. Selecciona un estilo visual
    3. Ajusta los parámetros (opcional)
    4. Haz clic en "Generar"
    """)
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="📝 Descripción del contenido",
                placeholder="Ej: Diagrama de célula vegetal con cloroplastos y núcleo etiquetados",
                lines=3
            )
            
            style_input = gr.Radio(
                choices=list(STYLES.keys()),
                value="Infografía",
                label="🎨 Estilo visual"
            )
            
            with gr.Row():
                steps_input = gr.Slider(
                    minimum=10, maximum=50, value=25, step=5,
                    label="Pasos"
                )
                guidance_input = gr.Slider(
                    minimum=1, maximum=15, value=7.5, step=0.5,
                    label="Guidance"
                )
            
            seed_input = gr.Number(
                value=-1,
                label="Semilla (-1 = aleatorio)"
            )
            
            with gr.Row():
                generate_btn = gr.Button("🚀 Generar", variant="primary")
                clear_btn = gr.Button("🗑️ Limpiar memoria")
        
        with gr.Column():
            output_image = gr.Image(label="Imagen generada", type="pil")
            output_msg = gr.Textbox(label="Estado", interactive=False)
    
    # Ejemplos
    gr.Examples(
        examples=[
            ["Diagrama de célula animal con núcleo, mitocondrias y membrana", "Científico", 25, 7.5, 42],
            ["Ciclo del agua con evaporación, condensación y precipitación", "Infografía", 25, 7.5, 123],
            ["Sistema solar con planetas etiquetados", "Ilustración", 25, 7.5, 456],
            ["Pirámide alimenticia con grupos de alimentos", "Diagrama", 25, 7.5, 789],
        ],
        inputs=[prompt_input, style_input, steps_input, guidance_input, seed_input],
    )
    
    # Eventos
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, style_input, steps_input, guidance_input, seed_input],
        outputs=[output_image, output_msg]
    )
    
    clear_btn.click(
        fn=clear_memory,
        outputs=[output_msg]
    )
    
    gr.Markdown("""
    ---
    ### ℹ️ Acerca de EduDiff
    
    **EduDiff** es un proyecto de IA generativa para crear material educativo visual.
    
    - **Modelo:** Stable Diffusion 1.5
    - **Uso:** Material didáctico, infografías, diagramas
    - **Nota:** El contenido generado debe ser verificado antes de su uso educativo
    
    ---
    *Proyecto EA3 - Generación de Contenido con IA Generativa*
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# INICIO
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 50)
print("🎓 EduDiff — Generador Educativo")
print(f"📍 Dispositivo: {_device.upper()}")
print(f"🔧 PyTorch: {torch.__version__}")
print("=" * 50)

if __name__ == "__main__":
    demo.launch()
