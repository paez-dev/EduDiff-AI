# ═══════════════════════════════════════════════════════════════════════════════
# EduDiff XL — Generador de Material Educativo con IA Generativa
# ═══════════════════════════════════════════════════════════════════════════════
# Proyecto: EA3 - Generación de Contenido con IA Generativa
# Arquitectura: Stable Diffusion XL + ControlNet (Modelos de Difusión)
# Dominio: Educación - Generación de infografías, diagramas y material didáctico
# ═══════════════════════════════════════════════════════════════════════════════

import os
import gc
import torch
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from diffusers import (
    ControlNetModel, 
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler
)
from datetime import datetime
import json

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════

SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# ControlNet SDXL verificados en Hugging Face
CONTROLNET_MODELS = {
    "Sin ControlNet": None,
    "Lineart (Contornos)": "xinsir/controlnet-lineart-sdxl-1.0",
    "Canny (Bordes)": "diffusers/controlnet-canny-sdxl-1.0",
    "Depth (Profundidad)": "diffusers/controlnet-depth-sdxl-1.0",
    "Sketch (Boceto)": "xinsir/controlnet-sketch-sdxl-1.0",
}

# Configuración por defecto
DEFAULT_STEPS = 25
DEFAULT_GUIDANCE = 7.5
MAX_IMAGE_SIDE = 1024

# Estado global (lazy loading para optimización de memoria)
_current_key = None
_current_pipe = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_generation_history = []

# ═══════════════════════════════════════════════════════════════════════════════
# ESTILOS EDUCATIVOS PREDEFINIDOS
# ═══════════════════════════════════════════════════════════════════════════════

EDUCATIONAL_STYLES = {
    "📊 Infografía Profesional": {
        "prompt": "professional infographic design, clean vector style, labeled diagram, white background, high contrast, modern educational material, clear typography",
        "negative": "blurry, low quality, text errors, cluttered, confusing layout"
    },
    "🎨 Ilustración Didáctica": {
        "prompt": "digital educational illustration, vibrant colors, soft shading, clear subject, child-friendly, engaging visual",
        "negative": "scary, dark, complex, abstract, photorealistic"
    },
    "✏️ Dibujo Escolar": {
        "prompt": "hand-drawn educational sketch, simple shapes, colorful crayons, fun classroom style, easy to understand",
        "negative": "realistic, dark colors, complex details, scary elements"
    },
    "🔬 Científico Detallado": {
        "prompt": "scientific illustration, anatomical detail, labeled parts, textbook quality, precise rendering, educational diagram",
        "negative": "cartoon, simplified, abstract, artistic interpretation"
    },
    "📐 Diagrama Técnico": {
        "prompt": "technical diagram, blueprint style, precise lines, measurements, engineering drawing, schematic view",
        "negative": "colorful, artistic, hand-drawn, sketch style"
    },
    "🌈 Mapa Conceptual": {
        "prompt": "concept map illustration, connected ideas, colorful nodes, flowing arrows, mind map style, organized layout",
        "negative": "realistic, photographic, single subject, no connections"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# PLANTILLAS DE PROMPTS POR ÁREA EDUCATIVA
# ═══════════════════════════════════════════════════════════════════════════════

EDUCATIONAL_TEMPLATES = {
    "🧬 Biología": [
        "Diagrama de célula {tipo} mostrando {orgánulos}, estilo educativo con etiquetas claras",
        "Ciclo de vida de {organismo} con flechas y etapas numeradas",
        "Sistema {sistema} humano con partes señaladas y funciones",
        "Cadena alimenticia de ecosistema {ecosistema} con niveles tróficos"
    ],
    "⚗️ Química": [
        "Tabla periódica interactiva destacando elementos {grupo}",
        "Modelo molecular de {molécula} en 3D con enlaces",
        "Reacción química de {reactivos} con productos y ecuación balanceada",
        "Estados de la materia con transiciones y ejemplos"
    ],
    "🔢 Matemáticas": [
        "Recta numérica mostrando {concepto} con ejemplos visuales",
        "Figuras geométricas {dimensión} con fórmulas de área y perímetro",
        "Gráfica de función {tipo} con ejes etiquetados",
        "Fracciones equivalentes representadas con círculos y rectángulos"
    ],
    "🌍 Geografía": [
        "Mapa de {región} con capitales y fronteras señaladas",
        "Ciclo del agua con etapas: evaporación, condensación, precipitación",
        "Capas de la Tierra con nombres y características",
        "Climas del mundo con iconos representativos"
    ],
    "📚 Historia": [
        "Línea del tiempo de {período} con eventos principales",
        "Mapa de {civilización} antigua con ciudades importantes",
        "Infografía de {evento histórico} con causas y consecuencias",
        "Pirámide social de {sociedad} con roles y jerarquías"
    ],
    "🌐 Idiomas": [
        "Vocabulario visual de {tema} en {idioma} con imágenes",
        "Conjugación de verbos {tipo} con ejemplos",
        "Mapa mental de gramática {tema} con reglas y excepciones",
        "Diálogo ilustrado en {idioma} sobre {situación}"
    ]
}

# ═══════════════════════════════════════════════════════════════════════════════
# GESTIÓN DE MEMORIA Y CARGA DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════════

def _unload_pipeline():
    """Libera la memoria del pipeline actual de forma segura."""
    global _current_key, _current_pipe
    
    if _current_pipe is None:
        return "No hay modelos cargados."
    
    try:
        _current_pipe.to("cpu")
    except:
        pass
    
    del _current_pipe
    _current_pipe = None
    _current_key = None
    
    if _device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    return "✅ Memoria liberada correctamente."


def _load_pipeline(controlnet_key: str):
    """
    Carga el pipeline de Stable Diffusion XL con ControlNet opcional.
    Implementa lazy loading para optimizar el uso de memoria.
    """
    global _current_key, _current_pipe
    
    # Verificar si ya está cargado
    if _current_key == controlnet_key and _current_pipe is not None:
        return _current_pipe
    
    # Descargar pipeline anterior
    if _current_pipe is not None:
        _unload_pipeline()
    
    dtype = torch.float16 if _device == "cuda" else torch.float32
    
    # Cargar sin ControlNet (más rápido)
    if controlnet_key == "Sin ControlNet":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if _device == "cuda" else None
        )
    else:
        # Cargar ControlNet específico
        cn_model_id = CONTROLNET_MODELS[controlnet_key]
        controlnet = ControlNetModel.from_pretrained(
            cn_model_id,
            torch_dtype=dtype,
            use_safetensors=True
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_MODEL,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if _device == "cuda" else None
        )
    
    # Optimizaciones
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    if _device == "cuda":
        pipe.to("cuda")
        # Habilitar optimizaciones de memoria si están disponibles
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    _current_key = controlnet_key
    _current_pipe = pipe
    
    return pipe


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE GENERACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_prompt(prompt: str) -> str:
    """Limpia y valida el prompt del usuario."""
    if not prompt or not prompt.strip():
        return "Educational infographic with clear labels and modern design"
    return prompt.strip()[:1500]


def generate_educational_image(
    prompt: str,
    style: str,
    steps: int,
    guidance_scale: float,
    controlnet_type: str,
    control_image,
    seed: int = -1,
    width: int = 1024,
    height: int = 1024
):
    """
    Genera una imagen educativa usando Stable Diffusion XL.
    
    Args:
        prompt: Descripción del contenido educativo a generar
        style: Estilo visual predefinido
        steps: Número de pasos de inferencia (más = mejor calidad)
        guidance_scale: Adherencia al prompt (7-9 recomendado)
        controlnet_type: Tipo de ControlNet para guiar la generación
        control_image: Imagen guía para ControlNet
        seed: Semilla para reproducibilidad (-1 = aleatorio)
        width: Ancho de la imagen
        height: Alto de la imagen
    
    Returns:
        tuple: (imagen_generada, mensaje_estado, info_generación)
    """
    global _generation_history
    
    # Preparar prompt con estilo
    clean_prompt = sanitize_prompt(prompt)
    style_config = EDUCATIONAL_STYLES.get(style, EDUCATIONAL_STYLES["📊 Infografía Profesional"])
    
    full_prompt = f"{clean_prompt}, {style_config['prompt']}"
    negative_prompt = style_config['negative'] + ", watermark, signature, text overlay, bad anatomy, deformed"
    
    # Validar parámetros
    steps = int(max(10, min(50, steps)))
    guidance_scale = float(max(1.0, min(15.0, guidance_scale)))
    width = int(min(1024, max(512, width)))
    height = int(min(1024, max(512, height)))
    
    # Configurar semilla
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device=_device).manual_seed(seed)
    
    # Procesar imagen de control si existe
    processed_control = None
    if control_image is not None and controlnet_type != "Sin ControlNet":
        if control_image.mode != "RGB":
            control_image = control_image.convert("RGB")
        processed_control = control_image.resize((width, height), Image.LANCZOS)
    
    # Cargar pipeline
    try:
        pipe = _load_pipeline(controlnet_type)
    except Exception as e:
        return None, f"❌ Error cargando modelo: {str(e)}", ""
    
    # Generar imagen
    try:
        gen_kwargs = {
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator
        }
        
        # Agregar imagen de control si aplica
        if processed_control is not None and controlnet_type != "Sin ControlNet":
            gen_kwargs["image"] = processed_control
            gen_kwargs["controlnet_conditioning_scale"] = 0.8
        
        if _device == "cuda":
            with torch.autocast("cuda"):
                result = pipe(**gen_kwargs)
        else:
            result = pipe(**gen_kwargs)
        
        generated_image = result.images[0]
        
    except Exception as e:
        _unload_pipeline()
        return None, f"❌ Error en generación: {str(e)}", ""
    
    # Registrar en historial
    generation_info = {
        "timestamp": datetime.now().isoformat(),
        "prompt": clean_prompt,
        "style": style,
        "steps": steps,
        "guidance": guidance_scale,
        "seed": seed,
        "controlnet": controlnet_type,
        "resolution": f"{width}x{height}"
    }
    _generation_history.append(generation_info)
    
    info_text = f"""### ✅ Generación Exitosa
    
**Semilla:** `{seed}` (guárdala para reproducir)

**Parámetros:**
- Pasos: {steps}
- Guidance: {guidance_scale}
- Resolución: {width}x{height}
- ControlNet: {controlnet_type}

**Prompt procesado:** {full_prompt[:200]}...
"""
    
    return generated_image, "✅ Imagen generada correctamente", info_text


def get_generation_stats():
    """Retorna estadísticas de las generaciones realizadas."""
    if not _generation_history:
        return "No hay generaciones registradas en esta sesión."
    
    total = len(_generation_history)
    styles_used = {}
    for gen in _generation_history:
        s = gen.get("style", "Unknown")
        styles_used[s] = styles_used.get(s, 0) + 1
    
    stats = f"""### 📊 Estadísticas de Sesión

**Total de generaciones:** {total}

**Estilos utilizados:**
"""
    for style, count in styles_used.items():
        stats += f"- {style}: {count}\n"
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE USUARIO - GRADIO
# ═══════════════════════════════════════════════════════════════════════════════

# CSS personalizado para una interfaz más profesional
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif;
}
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 1.5rem;
    color: white;
}
.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.main-header p {
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
}
.info-box {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-left: 4px solid #2d5a87;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.footer {
    text-align: center;
    padding: 1rem;
    color: #6c757d;
    font-size: 0.9rem;
}
"""

def create_interface():
    """Crea la interfaz Gradio completa."""
    
    with gr.Blocks(css=CUSTOM_CSS, title="EduDiff XL - Generador Educativo") as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎓 EduDiff XL</h1>
            <p>Generador de Material Educativo con Inteligencia Artificial</p>
            <p style="font-size: 0.9rem; margin-top: 0.5rem;">
                Powered by Stable Diffusion XL + ControlNet
            </p>
        </div>
        """)
        
        with gr.Tabs():
            # ═══════════════════════════════════════════════════════════════
            # TAB 1: GENERADOR PRINCIPAL
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("🎨 Generador"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Área de entrada
                        gr.Markdown("### 📝 Describe tu contenido educativo")
                        
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Ej: Diagrama de célula vegetal mostrando cloroplastos, vacuola central y pared celular con etiquetas claras",
                            lines=4,
                            max_lines=6
                        )
                        
                        with gr.Row():
                            style_dropdown = gr.Dropdown(
                                choices=list(EDUCATIONAL_STYLES.keys()),
                                value="📊 Infografía Profesional",
                                label="Estilo Visual"
                            )
                            controlnet_dropdown = gr.Dropdown(
                                choices=list(CONTROLNET_MODELS.keys()),
                                value="Sin ControlNet",
                                label="ControlNet"
                            )
                        
                        gr.Markdown("### ⚙️ Parámetros de Generación")
                        
                        with gr.Row():
                            steps_slider = gr.Slider(
                                minimum=10, maximum=50, value=25, step=1,
                                label="Pasos de Inferencia",
                                info="Más pasos = mejor calidad, más tiempo"
                            )
                            guidance_slider = gr.Slider(
                                minimum=1.0, maximum=15.0, value=7.5, step=0.5,
                                label="Guidance Scale",
                                info="Mayor = más fiel al prompt"
                            )
                        
                        with gr.Row():
                            width_slider = gr.Slider(
                                minimum=512, maximum=1024, value=1024, step=64,
                                label="Ancho"
                            )
                            height_slider = gr.Slider(
                                minimum=512, maximum=1024, value=1024, step=64,
                                label="Alto"
                            )
                        
                        seed_input = gr.Number(
                            value=-1, 
                            label="Semilla (-1 = aleatorio)",
                            precision=0
                        )
                        
                        control_image_input = gr.Image(
                            type="pil",
                            label="Imagen Guía (opcional para ControlNet)"
                        )
                        
                        with gr.Row():
                            generate_btn = gr.Button("🚀 Generar Imagen", variant="primary", size="lg")
                            clear_btn = gr.Button("🗑️ Limpiar", variant="secondary")
                    
                    with gr.Column(scale=1):
                        # Área de salida
                        gr.Markdown("### 🖼️ Resultado")
                        
                        output_image = gr.Image(
                            type="pil",
                            label="Imagen Generada",
                            height=512
                        )
                        
                        status_output = gr.Markdown("")
                        info_output = gr.Markdown("")
                        
                        with gr.Row():
                            download_btn = gr.Button("💾 Descargar")
                            unload_btn = gr.Button("🔄 Liberar Memoria")
                
                # Eventos
                generate_btn.click(
                    fn=generate_educational_image,
                    inputs=[
                        prompt_input, style_dropdown, steps_slider, 
                        guidance_slider, controlnet_dropdown, control_image_input,
                        seed_input, width_slider, height_slider
                    ],
                    outputs=[output_image, status_output, info_output]
                )
                
                clear_btn.click(
                    fn=lambda: (None, "", ""),
                    outputs=[output_image, status_output, info_output]
                )
                
                unload_btn.click(
                    fn=_unload_pipeline,
                    outputs=[status_output]
                )
            
            # ═══════════════════════════════════════════════════════════════
            # TAB 2: PLANTILLAS EDUCATIVAS
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("📚 Plantillas"):
                gr.Markdown("""
                ### 📖 Plantillas por Área Educativa
                
                Selecciona un área y una plantilla para obtener prompts optimizados.
                """)
                
                for area, templates in EDUCATIONAL_TEMPLATES.items():
                    with gr.Accordion(area, open=False):
                        for i, template in enumerate(templates):
                            gr.Markdown(f"**{i+1}.** `{template}`")
                
                gr.Markdown("""
                ---
                **💡 Tip:** Copia la plantilla y reemplaza los valores entre `{llaves}` con tu contenido específico.
                """)
            
            # ═══════════════════════════════════════════════════════════════
            # TAB 3: GALERÍA DE EJEMPLOS
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("🖼️ Galería"):
                gr.Markdown("""
                ### 🎯 Ejemplos de Contenido Generado
                
                Aquí se mostrarán ejemplos de las capacidades del sistema.
                
                **Casos de Uso:**
                
                1. **📚 Educación Primaria:** Diagramas simples del cuerpo humano, 
                   ciclos naturales, mapas conceptuales con colores vivos.
                
                2. **🔬 Educación Secundaria:** Ilustraciones científicas detalladas,
                   diagramas de química, mapas históricos.
                
                3. **🎓 Educación Superior:** Diagramas técnicos, infografías de 
                   investigación, visualizaciones de datos complejas.
                
                4. **👨‍🏫 Material Docente:** Presentaciones visuales, material 
                   de apoyo, evaluaciones ilustradas.
                """)
                
                stats_btn = gr.Button("📊 Ver Estadísticas de Sesión")
                stats_output = gr.Markdown("")
                
                stats_btn.click(fn=get_generation_stats, outputs=[stats_output])
            
            # ═══════════════════════════════════════════════════════════════
            # TAB 4: INFORMACIÓN DEL PROYECTO
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("ℹ️ Acerca de"):
                gr.Markdown("""
                ## 🎓 EduDiff XL — Proyecto EA3
                
                ### 📋 Descripción del Proyecto
                
                **EduDiff** es una aplicación de inteligencia artificial generativa diseñada para 
                crear material educativo visual de alta calidad. Utiliza modelos de difusión 
                de última generación (Stable Diffusion XL) combinados con ControlNet para 
                ofrecer control preciso sobre la generación de imágenes.
                
                ### 🎯 Problema que Resuelve
                
                Los docentes y creadores de contenido educativo enfrentan desafíos para:
                - Crear material visual atractivo y didáctico
                - Personalizar ilustraciones para necesidades específicas
                - Generar contenido rápidamente sin habilidades de diseño
                
                ### 🛠️ Tecnologías Utilizadas
                
                | Componente | Tecnología |
                |------------|------------|
                | Modelo Base | Stable Diffusion XL 1.0 |
                | Control Adicional | ControlNet (Canny, Depth, Lineart, Sketch) |
                | Scheduler | DPM++ Solver |
                | Framework | Diffusers (Hugging Face) |
                | Interfaz | Gradio |
                | Optimización | xFormers, FP16 |
                
                ### 👥 Usuarios Finales
                
                - **Docentes** de todos los niveles educativos
                - **Diseñadores instruccionales** 
                - **Creadores de contenido educativo**
                - **Estudiantes** para proyectos y presentaciones
                
                ### ⚖️ Consideraciones Éticas
                
                Este sistema incluye:
                - Filtros de contenido para evitar generación inapropiada
                - Transparencia sobre el origen IA del contenido
                - Recomendaciones de uso responsable
                
                ---
                
                **Versión:** 1.0.0 | **Licencia:** Educativa
                """)
        
        # Footer
        gr.HTML("""
        <div class="footer">
            <p>🎓 EduDiff XL — Generación de Contenido Educativo con IA</p>
            <p>Proyecto EA3 - Inteligencia Artificial Generativa</p>
        </div>
        """)
    
    return demo


# ═══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🎓 EduDiff XL — Generador de Material Educativo")
    print("=" * 60)
    print(f"📍 Dispositivo: {_device.upper()}")
    print(f"🔧 PyTorch: {torch.__version__}")
    print("=" * 60)
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
