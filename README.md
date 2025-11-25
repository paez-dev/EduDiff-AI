# EduDiff — Generador de Imágenes Educativas (EA3)

## Resumen
EduDiff genera imágenes educativas (infografías, diagramas, esquemas) usando Stable Diffusion con ajuste LoRA. Proporciona un notebook reproducible, una app Gradio para demo y documentación lista para entrega.

## Archivos entregados
- `Apellido1_Apellido2_Apellido3_EA3_GenerativeAI.pdf` — Informe (4 páginas).
- `Apellido1_Apellido2_Apellido3_EA3_GenerativeAI_Notebook.ipynb` — Colab notebook.
- `app/app.py` — App Gradio.
- `data/` — Instrucciones para preparar dataset (no incluir imágenes con copyright).
- `results/` — Imágenes generadas y métricas (FID).
- `video/` — Video de presentación (.mp4).
- `README.md` — Este archivo.

## Requisitos
- Python 3.10+, GPU (recomendado).
- Dependencias principales:
  - diffusers, transformers, accelerate, peft, safetensors, gradio, pytorch-fid

## Cómo ejecutar (Colab)
1. Abrir `Apellido1_Apellido2_Apellido3_EA3_GenerativeAI_Notebook.ipynb` en Google Colab.  
2. Seleccionar runtime GPU.  
3. Ejecutar celdas en orden.  
4. (Opcional) Ejecutar entrenamiento LoRA si tienes GPU potente; si no, usar modelo base y ajustar prompts.

## Cómo desplegar la app (Gradio)
1. Editar `app/app.py` para apuntar al modelo y, si aplica, cargar LoRA.  
2. Ejecutar localmente: `python app.py` (asegúrate de tener CUDA y dependencias).  
3. Para despliegue público: usar Hugging Face Spaces (si tu modelo y datos pueden ser públicos) o desplegar en una VM/servicio con GPU.

## Evaluación y métricas
- FID con `pytorch-fid` comparando carpetas `/real` y `/gen`.  
- Encuesta humana (Google Forms) para docentes.

## Ética y licencias
- Evitar usar imágenes con copyright; preferir CC0/CC BY.  
- Incluir disclaimers y watermark si corresponde.

## Referencia al enunciado de la actividad
Enunciado original subido por el curso: `/mnt/data/EA3.docx`. :contentReference[oaicite:1]{index=1}