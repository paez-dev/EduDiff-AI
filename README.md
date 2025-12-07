---
title: EduDiff XL
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
license: mit
---

# 🎓 EduDiff XL — Generador de Material Educativo con IA Generativa

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44+-orange.svg)](https://gradio.app/)

## 📋 Descripción

**EduDiff XL** es una aplicación de inteligencia artificial generativa diseñada para crear material educativo visual de alta calidad. Utiliza modelos de difusión de última generación (Stable Diffusion XL) combinados con ControlNet para ofrecer control preciso sobre la generación de imágenes.

### 🎯 Problema que Resuelve

Los docentes y creadores de contenido educativo enfrentan desafíos para:
- Crear material visual atractivo y didáctico sin habilidades de diseño
- Personalizar ilustraciones para necesidades específicas del aula
- Generar contenido rápidamente manteniendo calidad profesional

### ✨ Características Principales

- **🖼️ Generación de alta calidad** con Stable Diffusion XL (1024x1024)
- **🎨 Múltiples estilos educativos**: Infografía, Ilustración, Científico, Diagrama
- **🔧 ControlNet integrado** para control de composición (Canny, Depth, Lineart, Sketch)
- **📚 Plantillas por área**: Biología, Química, Matemáticas, Geografía, Historia
- **⚡ Optimizado para GPU** con soporte para CPU

---

## 🚀 Instalación

### Requisitos del Sistema

- Python 3.10 o superior
- GPU con 8GB+ VRAM (recomendado) o CPU (más lento)
- 15GB de espacio en disco

### Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/EduDiff-AI.git
cd EduDiff-AI

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
python app.py
```

### Ejecución en Google Colab

1. Abrir el notebook `notebooks/EA3_EduDiff_Notebook.ipynb` en Google Colab
2. Ejecutar las celdas en orden
3. La interfaz Gradio se abrirá automáticamente con un enlace público

---

## 📖 Guía de Usuario

### 1. Inicio Rápido

1. **Abrir la aplicación** en el navegador (localhost:7860 o enlace público)
2. **Escribir un prompt** describiendo el contenido educativo deseado
3. **Seleccionar el estilo** visual apropiado para tu audiencia
4. **Ajustar parámetros** (opcional):
   - **Steps**: 25-35 para balance calidad/velocidad
   - **Guidance**: 7-9 para mejor adherencia al prompt
5. **Generar** y descargar la imagen

### 2. Mejores Prácticas para Prompts

```
✅ BUENOS EJEMPLOS:
- "Diagrama de célula vegetal mostrando cloroplastos, vacuola central y pared celular con etiquetas claras"
- "Infografía del ciclo del agua con evaporación, condensación y precipitación, flechas y etiquetas"
- "Sistema solar con planetas a escala, nombres y órbitas visibles"

❌ EVITAR:
- Prompts muy cortos: "célula"
- Prompts ambiguos: "algo educativo"
- Prompts sin contexto: "diagrama"
```

### 3. Estilos Disponibles

| Estilo | Uso Recomendado | Audiencia |
|--------|-----------------|-----------|
| 📊 Infografía | Presentaciones, material impreso | General |
| 🎨 Ilustración | Material para niños | Primaria |
| 🔬 Científico | Textos académicos | Secundaria/Superior |
| 📐 Diagrama | Procesos y sistemas | Técnico |

---

## 🛠️ Arquitectura Técnica

```
EduDiff-AI/
├── app.py                 # Aplicación principal Gradio
├── requirements.txt       # Dependencias
├── README.md             # Documentación
├── notebooks/
│   └── EA3_EduDiff_Notebook.ipynb  # Notebook completo
├── src/
│   └── utils.py          # Utilidades
└── results/
    ├── experiments/      # Resultados de experimentos
    ├── metrics/         # Métricas de evaluación
    └── portfolio/       # Ejemplos generados
    └── experiments.md       # Explicación de lo que se hizo

```

### Tecnologías Utilizadas

| Componente | Tecnología |
|------------|------------|
| Modelo Base | Stable Diffusion XL 1.0 |
| Control | ControlNet (Canny, Depth, Lineart, Sketch) |
| Scheduler | DPM++ Solver Multistep |
| Framework | Diffusers (Hugging Face) |
| Interfaz | Gradio 4.44+ |
| Optimización | xFormers, FP16 |

---

## 📊 Experimentación

Se realizaron 3 experimentos principales:

1. **Variación de Guidance Scale** (3.0, 7.5, 12.0)
   - Resultado: 7.5 ofrece mejor balance creatividad/fidelidad

2. **Variación de Inference Steps** (15, 30, 50)
   - Resultado: 25-35 steps óptimos para producción

3. **Comparación de Estilos Educativos**
   - Resultado: Cada estilo tiene audiencia específica

Ver detalles en el notebook.

---

## ⚖️ Consideraciones Éticas

### Sesgos Identificados
- Sub-representación de contextos latinoamericanos
- Posible sesgo en representación de diversidad

### Mitigación
1. Prompt engineering consciente
2. Revisión humana antes del uso
3. Guías de uso responsable

### Uso Responsable
- ✅ Verificar precisión del contenido antes del uso educativo
- ✅ Indicar que el contenido es generado por IA
- ❌ No usar para crear contenido engañoso o inapropiado

---

## 👥 Casos de Uso

1. **Docentes de Primaria**: Diagramas coloridos para ciencias naturales
2. **Profesores de Secundaria**: Ilustraciones científicas detalladas
3. **Diseñadores Instruccionales**: Material visual para cursos online
4. **Autores Educativos**: Ilustraciones para libros de texto

---

## 📄 Licencia

Este proyecto es para uso educativo. Ver LICENSE para más detalles.

---

## 🙏 Agradecimientos

- [Stability AI](https://stability.ai/) por Stable Diffusion XL
- [Hugging Face](https://huggingface.co/) por Diffusers
- [Gradio](https://gradio.app/) por la interfaz

---

**EA3 — Generación de Contenido con IA Generativa**
