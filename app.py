"""
═══════════════════════════════════════════════════════════════════════════════
EduDiff XL — Generador de Material Educativo con IA Generativa
═══════════════════════════════════════════════════════════════════════════════
Proyecto: EA3 - Generación de Contenido con IA Generativa
Arquitectura: Modelos de Difusión (Stable Diffusion / FLUX)
Dominio: Educación - Generación de infografías, diagramas y material didáctico
═══════════════════════════════════════════════════════════════════════════════
"""

import gradio as gr
from gradio_client import Client
from PIL import Image
import io
import base64

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

ESTILOS = {
    "📊 Infografía Profesional": "professional infographic, clean vector design, labeled diagram, white background, high contrast, modern educational material",
    "🎨 Ilustración Didáctica": "digital educational illustration, vibrant colors, child-friendly, engaging visual, cartoon style",
    "🔬 Científico Detallado": "scientific illustration, anatomical detail, textbook quality, precise rendering, labeled parts",
    "📐 Diagrama Técnico": "technical diagram, blueprint style, precise lines, schematic view, engineering drawing",
    "✏️ Dibujo Escolar": "hand-drawn sketch, simple shapes, colorful, classroom style, easy to understand",
    "🌈 Mapa Conceptual": "concept map, connected ideas, colorful nodes, mind map style, organized layout"
}

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE GENERACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def generar_imagen(prompt: str, estilo: str, calidad: str) -> tuple:
    """
    Genera una imagen educativa usando modelos de difusión.
    
    Args:
        prompt: Descripción del contenido educativo
        estilo: Estilo visual seleccionado
        calidad: Nivel de calidad (Rápida/Estándar/Alta)
    
    Returns:
        tuple: (imagen, mensaje de estado)
    """
    if not prompt or not prompt.strip():
        return None, "⚠️ Por favor, ingresa una descripción del contenido educativo."
    
    # Construir prompt completo
    estilo_prompt = ESTILOS.get(estilo, ESTILOS["📊 Infografía Profesional"])
    prompt_completo = f"{prompt}, {estilo_prompt}, high quality, detailed"
    
    # Configurar pasos según calidad
    steps_map = {"⚡ Rápida": 20, "⭐ Estándar": 30, "💎 Alta": 40}
    num_steps = steps_map.get(calidad, 30)
    
    try:
        # Usar cliente de API pública de HuggingFace
        client = Client("black-forest-labs/FLUX.1-schnell")
        
        result = client.predict(
            prompt=prompt_completo,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            num_inference_steps=4,
            api_name="/infer"
        )
        
        # El resultado es una tupla (imagen_path, seed)
        if result and len(result) > 0:
            image_path = result[0]
            seed_used = result[1] if len(result) > 1 else "N/A"
            return image_path, f"✅ Imagen generada exitosamente\n📌 Semilla: {seed_used}"
        else:
            return None, "❌ No se pudo generar la imagen"
            
    except Exception as e:
        error_msg = str(e)
        if "exceeded" in error_msg.lower() or "limit" in error_msg.lower():
            return None, "⏳ Límite de API alcanzado. Espera unos segundos e intenta de nuevo."
        elif "loading" in error_msg.lower():
            return None, "🔄 El modelo se está cargando. Espera 30 segundos e intenta de nuevo."
        else:
            return None, f"❌ Error: {error_msg[:150]}"

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE USUARIO
# ═══════════════════════════════════════════════════════════════════════════════

# CSS personalizado
css = """
.gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif;
}
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}
.main-title h1 {
    color: white;
    margin: 0;
    font-size: 2rem;
}
.main-title p {
    color: rgba(255,255,255,0.9);
    margin: 0.5rem 0 0 0;
}
footer {visibility: hidden}
"""

# Crear interfaz
with gr.Blocks(css=css, title="EduDiff XL", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.HTML("""
    <div class="main-title">
        <h1>🎓 EduDiff XL</h1>
        <p>Generador de Material Educativo con Inteligencia Artificial</p>
    </div>
    """)
    
    with gr.Row():
        # Panel izquierdo - Controles
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Configuración")
            
            prompt_input = gr.Textbox(
                label="Descripción del contenido",
                placeholder="Ej: Diagrama de célula vegetal mostrando cloroplastos, vacuola central, pared celular y núcleo con etiquetas claras",
                lines=4,
                max_lines=6
            )
            
            estilo_input = gr.Dropdown(
                choices=list(ESTILOS.keys()),
                value="📊 Infografía Profesional",
                label="Estilo visual"
            )
            
            calidad_input = gr.Radio(
                choices=["⚡ Rápida", "⭐ Estándar", "💎 Alta"],
                value="⭐ Estándar",
                label="Calidad"
            )
            
            generar_btn = gr.Button("🚀 Generar Imagen", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            ### 💡 Consejos
            - Sé específico en tu descripción
            - Menciona qué elementos deben etiquetarse
            - Indica el nivel educativo si es relevante
            """)
        
        # Panel derecho - Resultado
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Resultado")
            
            output_image = gr.Image(
                label="Imagen Generada",
                type="filepath",
                height=450
            )
            
            status_output = gr.Textbox(
                label="Estado",
                interactive=False,
                lines=2
            )
    
    # Ejemplos
    gr.Markdown("### 📚 Ejemplos de uso")
    gr.Examples(
        examples=[
            ["Diagrama de célula animal con núcleo, mitocondrias, ribosomas y membrana celular etiquetados", "🔬 Científico Detallado", "⭐ Estándar"],
            ["Ciclo del agua mostrando evaporación, condensación, precipitación con flechas", "📊 Infografía Profesional", "⭐ Estándar"],
            ["Sistema solar con los 8 planetas en orden, con nombres y tamaños relativos", "🎨 Ilustración Didáctica", "⭐ Estándar"],
            ["Pirámide alimenticia con grupos de alimentos y porciones recomendadas", "📊 Infografía Profesional", "⭐ Estándar"],
            ["Anatomía del corazón humano con aurículas, ventrículos y válvulas", "🔬 Científico Detallado", "💎 Alta"],
        ],
        inputs=[prompt_input, estilo_input, calidad_input],
        cache_examples=False
    )
    
    # Footer
    gr.Markdown("""
    ---
    <center>
    
    **EduDiff XL** — Proyecto EA3: Generación de Contenido con IA Generativa
    
    Modelo: FLUX.1-schnell | Framework: Gradio
    
    ⚠️ El contenido generado debe ser verificado antes de su uso educativo
    
    </center>
    """)
    
    # Evento de generación
    generar_btn.click(
        fn=generar_imagen,
        inputs=[prompt_input, estilo_input, calidad_input],
        outputs=[output_image, status_output]
    )

# ═══════════════════════════════════════════════════════════════════════════════
# INICIO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch()
