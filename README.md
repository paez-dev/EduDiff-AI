---
title: EduDiff EA3
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.0.1"
app_file: app.py
pinned: false
---

# EduDiff — Generador de Imágenes Educativas (EA3)

Esta aplicación genera imágenes educativas (infografías, diagramas y esquemas) usando Stable Diffusion.

## Cómo usar el Space

1. Abrir el enlace público de Hugging Face Spaces.
2. Introducir un **prompt** descriptivo en el cuadro de texto.
3. Seleccionar el **estilo** (Infografía, Ilustración, Dibujo escolar, Realista suave).
4. Ajustar los parámetros:
   - Num Inference Steps: controla la calidad y detalle.
   - Guidance Scale: controla cuán fiel es la imagen al prompt.
5. Presionar **Submit / Generar** y esperar la imagen generada.

## Requisitos

- Hugging Face Spaces se encarga de instalar dependencias automáticamente usando `requirements.txt`.
- Si quieres ejecutar localmente:
  ```bash
  pip install -r requirements.txt
  python app.py