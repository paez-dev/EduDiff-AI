# app.py - EduDiff XL (SDXL + ControlNet Suite) - Lazy loading / single-model-in-memory
import os
import gc
import torch
import gradio as gr
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

# -----------------------
# CONFIG – modelos SDXL y ControlNet reales
# -----------------------
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# ControlNet SDXL que EXISTEN en Hugging Face
CONTROLNET_MODELS = {
    "None": None,  # opción sin controlnet
    "Lineart": "xinsir/controlnet-lineart-sdxl-1.0",
    "Canny": "diffusers/controlnet-canny-sdxl-1.0",
    "Depth": "diffusers/controlnet-depth-sdxl-1.0",
    "Sketch": "xinsir/controlnet-sketch-sdxl-1.0",
}

controlnet_choices = list(CONTROLNET_MODELS.keys())

# Ajustes por defecto
DEFAULT_STEPS = 25
DEFAULT_GUIDANCE = 7.0
MAX_IMAGE_SIDE = 1024  # limitar tamaño

# Globals (lazy loading)
_current_key = None
_current_pipe = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Utilities: carga y descarga
# -----------------------
def _unload_current_pipeline():
    global _current_key, _current_pipe

    if _current_pipe is None:
        return

    try:
        try:
            _current_pipe.to("cpu")
        except:
            pass
        del _current_pipe
    except:
        pass

    _current_pipe = None
    _current_key = None

    if _device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def _load_controlnet(controlnet_key):
    global _current_key, _current_pipe

    # Ya cargado
    if _current_key == controlnet_key and _current_pipe is not None:
        return _current_pipe

    # Descargar anterior
    if _current_pipe is not None:
        _unload_current_pipeline()

    # Cargar ControlNet si aplica
    if controlnet_key == "None":
        cn_model = None
    else:
        model_id = CONTROLNET_MODELS[controlnet_key]
        cn_model = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            use_safetensors=True
        )

    # Cargar pipeline SDXL + ControlNet
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_MODEL,
        controlnet=cn_model,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        use_safetensors=True,
    )

    if _device == "cuda":
        pipe.to("cuda")

    _current_key = controlnet_key
    _current_pipe = pipe

    return pipe


# -----------------------
# Estilos
# -----------------------
STYLE_PROMPTS = {
    "Infografía": "flat infographic design, clear readable labels, vector style, white background, high contrast",
    "Ilustración": "digital illustration, clean colors, soft shading, clear subject",
    "Dibujo escolar": "child-like drawing, simple shapes, high contrast labels, crayon style",
    "Realista suave": "soft photorealistic rendering, natural lighting, subtle colors"
}


def _sanitize_prompt(prompt: str) -> str:
    if not prompt:
        return "Educational infographic template"
    p = prompt.strip()
    return p[:1000]


# -----------------------
# Función principal
# -----------------------
def generar(prompt, estilo, steps, scale, tipo_controlnet, input_image):
    prompt = _sanitize_prompt(prompt)
    final_prompt = f"{prompt}, {STYLE_PROMPTS.get(estilo, '')}"

    steps = int(max(1, min(80, steps)))
    scale = float(max(1.0, min(20.0, scale)))

    # Procesar imagen guía
    img_for_control = None
    if input_image is not None:
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        w, h = input_image.size
        max_side = max(w, h)

        if max_side > MAX_IMAGE_SIDE:
            s = MAX_IMAGE_SIDE / max_side
            input_image = input_image.resize((int(w * s), int(h * s)), Image.LANCZOS)

        img_for_control = input_image

    # Cargar pipeline
    try:
        pipe = _load_controlnet(tipo_controlnet)
    except Exception as e:
        return None, f"Error cargando modelo ControlNet '{tipo_controlnet}': {e}"

    # Generar
    try:
        if _device == "cuda":
            with torch.autocast("cuda"):
                result = pipe(
                    prompt=final_prompt,
                    control_image=img_for_control,
                    controlnet_conditioning_scale=1.0 if tipo_controlnet != "None" else 0.0,
                    num_inference_steps=steps,
                    guidance_scale=scale,
                )
        else:
            result = pipe(
                prompt=final_prompt,
                control_image=img_for_control,
                controlnet_conditioning_scale=1.0 if tipo_controlnet != "None" else 0.0,
                num_inference_steps=steps,
                guidance_scale=scale,
            )
    except Exception as e:
        _unload_current_pipeline()
        return None, f"Error durante generación: {e}"

    img = result.images[0]

    _unload_current_pipeline()

    return img, "OK"


# -----------------------
# Interfaz Gradio
# -----------------------
prompt_box = gr.Textbox(label="Prompt", lines=3)
style_box = gr.Dropdown(list(STYLE_PROMPTS.keys()), value="Infografía", label="Estilo")
steps_box = gr.Slider(5, 80, value=DEFAULT_STEPS, step=1, label="Num Steps")
scale_box = gr.Slider(1.0, 20.0, value=DEFAULT_GUIDANCE, step=0.1, label="Guidance Scale")
controlnet_box = gr.Dropdown(controlnet_choices, value="None", label="ControlNet")
image_input = gr.Image(type="pil", label="Imagen guía (opcional)")
output_img = gr.Image(type="pil", label="Resultado")
output_msg = gr.Markdown(label="Mensajes")

iface = gr.Interface(
    fn=generar,
    inputs=[prompt_box, style_box, steps_box, scale_box, controlnet_box, image_input],
    outputs=[output_img, output_msg],
    title="EduDiff XL — SDXL + ControlNet (Modelos Reales)",
    description="Se cargan únicamente ControlNet SDXL verificados y reales de HuggingFace."
)

unload_btn = gr.Button("Descargar modelos (liberar memoria)")

def unload_button_fn():
    _unload_current_pipeline()
    return "Modelos descargados."

unload_btn.click(fn=unload_button_fn, inputs=[], outputs=output_msg)

demo = gr.Column([iface, unload_btn])

if __name__ == "__main__":
    demo.launch(share=False)