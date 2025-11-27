"""
═══════════════════════════════════════════════════════════════════════════════
EduDiff XL — Generador de Material Educativo con IA Generativa
═══════════════════════════════════════════════════════════════════════════════
Proyecto: EA3 - Generación de Contenido con IA Generativa
Arquitectura: Modelos de Difusión (Stable Diffusion XL)
Dominio: Educación - Generación de infografías, diagramas y material didáctico
═══════════════════════════════════════════════════════════════════════════════
"""

import gradio as gr
from gradio_client import Client

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

ESTILOS = {
    "📊 Infografía Profesional": "professional infographic, clean vector design, labeled diagram, white background, high contrast, modern educational material, sharp details",
    "🎨 Ilustración Didáctica": "digital educational illustration, vibrant colors, child-friendly, engaging visual, cartoon style, clear shapes",
    "🔬 Científico Detallado": "scientific illustration, anatomical detail, textbook quality, precise rendering, labeled parts, medical illustration style",
    "📐 Diagrama Técnico": "technical diagram, blueprint style, precise lines, schematic view, engineering drawing, measurements",
    "✏️ Dibujo Escolar": "hand-drawn sketch, simple shapes, colorful crayons, classroom style, easy to understand, friendly",
    "🌈 Mapa Conceptual": "concept map, connected ideas, colorful nodes, mind map style, organized layout, arrows and connections"
}

NEGATIVE_PROMPT = "blurry, bad quality, distorted, ugly, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, cropped, worst quality, low quality, text errors, watermark, signature"

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE GENERACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def generar_imagen(prompt: str, estilo: str, guidance_scale: float, num_steps: int, seed: int) -> tuple:
    """
    Genera una imagen educativa usando Stable Diffusion XL.
    
    Args:
        prompt: Descripción del contenido educativo
        estilo: Estilo visual seleccionado
        guidance_scale: Control de adherencia al prompt (1-20)
        num_steps: Número de pasos de inferencia (10-50)
        seed: Semilla para reproducibilidad (-1 = aleatorio)
    
    Returns:
        tuple: (imagen, mensaje de estado)
    """
    if not prompt or not prompt.strip():
        return None, "⚠️ Por favor, ingresa una descripción del contenido educativo."
    
    # Construir prompt completo
    estilo_prompt = ESTILOS.get(estilo, ESTILOS["📊 Infografía Profesional"])
    prompt_completo = f"{prompt}, {estilo_prompt}, masterpiece, best quality, highly detailed"
    
    # Semilla
    use_random = seed == -1
    actual_seed = seed if seed >= 0 else 0
    
    try:
        # Usar Stable Diffusion 3.5 via Space público de Stability AI
        client = Client("stabilityai/stable-diffusion-3.5-large")
        
        result = client.predict(
            prompt=prompt_completo,
            negative_prompt=NEGATIVE_PROMPT,
            seed=actual_seed,
            randomize_seed=use_random,
            width=1024,
            height=1024,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            api_name="/infer"
        )
        
        if result:
            # El resultado puede ser una imagen o tupla
            if isinstance(result, tuple):
                image_path = result[0]
                used_seed = result[1] if len(result) > 1 else actual_seed
            else:
                image_path = result
                used_seed = actual_seed
            
            return image_path, f"✅ Generado | Guidance: {guidance_scale} | Steps: {num_steps} | Seed: {used_seed}"
        else:
            return None, "❌ No se pudo generar la imagen"
            
    except Exception as e:
        error_msg = str(e)
        if "exceeded" in error_msg.lower() or "limit" in error_msg.lower() or "queue" in error_msg.lower():
            return None, "⏳ Servidor ocupado. Espera unos segundos e intenta de nuevo."
        elif "loading" in error_msg.lower():
            return None, "🔄 El modelo se está cargando. Espera 30-60 segundos e intenta de nuevo."
        else:
            return None, f"❌ Error: {error_msg[:200]}"

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE USUARIO
# ═══════════════════════════════════════════════════════════════════════════════

with gr.Blocks() as demo:
    
    # Header
    gr.Markdown("""
    # 🎓 EduDiff XL
    ### Generador de Material Educativo con Inteligencia Artificial
    
    Crea imágenes educativas de alta calidad usando **Stable Diffusion 3.5**.
    """)
    
    with gr.Row():
        # Panel izquierdo - Controles
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Configuración")
            
            prompt_input = gr.Textbox(
                label="Descripción del contenido",
                placeholder="Ej: Diagrama de célula vegetal mostrando cloroplastos, vacuola central, pared celular y núcleo con etiquetas claras",
                lines=4
            )
            
            estilo_input = gr.Dropdown(
                choices=list(ESTILOS.keys()),
                value="📊 Infografía Profesional",
                label="Estilo visual"
            )
            
            gr.Markdown("### ⚙️ Parámetros")
            
            guidance_input = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=7.5,
                step=0.5,
                label="Guidance Scale (adherencia al prompt)",
                info="Bajo (1-5): más creativo | Medio (6-9): balanceado | Alto (10+): más literal"
            )
            
            steps_input = gr.Slider(
                minimum=10,
                maximum=50,
                value=25,
                step=5,
                label="Inference Steps (calidad)",
                info="Más pasos = mejor calidad pero más lento"
            )
            
            seed_input = gr.Number(
                value=-1,
                label="Seed (-1 = aleatorio)",
                precision=0
            )
            
            generar_btn = gr.Button("🚀 Generar Imagen", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            ### 💡 Consejos
            - **Guidance 7-9**: Balance óptimo para contenido educativo
            - **Steps 25-35**: Buena calidad sin esperar mucho
            - Guarda el **seed** para reproducir resultados
            """)
        
        # Panel derecho - Resultado
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Resultado")
            
            output_image = gr.Image(
                label="Imagen Generada",
                type="filepath",
                height=500
            )
            
            status_output = gr.Textbox(
                label="Estado",
                interactive=False
            )
    
    # Ejemplos
    gr.Markdown("### 📚 Ejemplos de uso")
    gr.Examples(
        examples=[
            ["Diagrama de célula animal con núcleo, mitocondrias, ribosomas y membrana celular etiquetados", "🔬 Científico Detallado", 7.5, 30, -1],
            ["Ciclo del agua mostrando evaporación, condensación, precipitación con flechas y etiquetas", "📊 Infografía Profesional", 8.0, 25, -1],
            ["Sistema solar con los 8 planetas en orden, con nombres y tamaños relativos", "🎨 Ilustración Didáctica", 7.0, 25, -1],
            ["Pirámide alimenticia con grupos de alimentos y porciones recomendadas", "📊 Infografía Profesional", 7.5, 25, -1],
            ["Anatomía del corazón humano con aurículas, ventrículos y válvulas etiquetados", "🔬 Científico Detallado", 8.5, 35, -1],
        ],
        inputs=[prompt_input, estilo_input, guidance_input, steps_input, seed_input],
        cache_examples=False
    )
    
    # Footer
    gr.Markdown("""
    ---
    **EduDiff XL** — Proyecto EA3: Generación de Contenido con IA Generativa
    
    Modelo: Stable Diffusion 3.5 | ⚠️ Verificar contenido antes de uso educativo
    """)
    
    # Evento de generación
    generar_btn.click(
        fn=generar_imagen,
        inputs=[prompt_input, estilo_input, guidance_input, steps_input, seed_input],
        outputs=[output_image, status_output]
    )

# ═══════════════════════════════════════════════════════════════════════════════
# INICIO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch()
