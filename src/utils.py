# ═══════════════════════════════════════════════════════════════════════════════
# EduDiff XL — Utilidades y Funciones Auxiliares
# ═══════════════════════════════════════════════════════════════════════════════

import os
import gc
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json


def get_device() -> str:
    """Detecta y retorna el dispositivo disponible (cuda o cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def clear_memory():
    """Libera memoria GPU y ejecuta garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_generation_metadata(
    image_path: str,
    metadata: Dict,
    output_dir: str = "results"
) -> str:
    """
    Guarda los metadatos de una generación en un archivo JSON.
    
    Args:
        image_path: Ruta de la imagen generada
        metadata: Diccionario con metadatos de la generación
        output_dir: Directorio de salida
    
    Returns:
        Ruta del archivo JSON guardado
    """
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.splitext(image_path)[0] + "_metadata.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return json_path


def calculate_image_quality_score(image: Image.Image) -> Dict[str, float]:
    """
    Calcula métricas de calidad básicas para una imagen.
    
    Args:
        image: Imagen PIL
    
    Returns:
        Diccionario con scores de calidad
    """
    img_array = np.array(image.convert('RGB'))
    
    # Brillo promedio (0-1)
    brightness = np.mean(img_array) / 255.0
    
    # Contraste (desviación estándar normalizada)
    contrast = np.std(img_array) / 255.0
    
    # Saturación promedio
    hsv = np.array(image.convert('HSV'))
    saturation = np.mean(hsv[:, :, 1]) / 255.0
    
    # Score de claridad (basado en contraste)
    clarity = min(1.0, contrast * 2.5)
    
    # Score general
    overall = (clarity * 0.4 + saturation * 0.3 + (1 - abs(brightness - 0.5)) * 0.3)
    
    return {
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "saturation": round(saturation, 3),
        "clarity": round(clarity, 3),
        "overall_score": round(overall, 3)
    }


def resize_image_for_controlnet(
    image: Image.Image,
    target_width: int = 1024,
    target_height: int = 1024
) -> Image.Image:
    """
    Redimensiona una imagen para uso con ControlNet.
    
    Args:
        image: Imagen PIL de entrada
        target_width: Ancho objetivo
        target_height: Alto objetivo
    
    Returns:
        Imagen redimensionada
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image.resize((target_width, target_height), Image.LANCZOS)


def validate_prompt(prompt: str) -> Tuple[bool, str]:
    """
    Valida un prompt de entrada.
    
    Args:
        prompt: Texto del prompt
    
    Returns:
        Tuple (es_válido, mensaje)
    """
    if not prompt or not prompt.strip():
        return False, "El prompt no puede estar vacío"
    
    if len(prompt) < 10:
        return False, "El prompt debe tener al menos 10 caracteres"
    
    if len(prompt) > 2000:
        return False, "El prompt no debe exceder 2000 caracteres"
    
    return True, "Prompt válido"


def create_image_grid(
    images: List[Image.Image],
    rows: int = 2,
    cols: int = 3,
    padding: int = 10,
    bg_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Crea una cuadrícula de imágenes.
    
    Args:
        images: Lista de imágenes PIL
        rows: Número de filas
        cols: Número de columnas
        padding: Espacio entre imágenes
        bg_color: Color de fondo RGB
    
    Returns:
        Imagen de la cuadrícula
    """
    if not images:
        raise ValueError("La lista de imágenes no puede estar vacía")
    
    # Obtener tamaño de la primera imagen
    img_width, img_height = images[0].size
    
    # Calcular tamaño total
    grid_width = cols * img_width + (cols + 1) * padding
    grid_height = rows * img_height + (rows + 1) * padding
    
    # Crear imagen de fondo
    grid = Image.new('RGB', (grid_width, grid_height), bg_color)
    
    # Pegar imágenes
    for idx, img in enumerate(images[:rows * cols]):
        row = idx // cols
        col = idx % cols
        
        x = padding + col * (img_width + padding)
        y = padding + row * (img_height + padding)
        
        # Redimensionar si es necesario
        if img.size != (img_width, img_height):
            img = img.resize((img_width, img_height), Image.LANCZOS)
        
        grid.paste(img, (x, y))
    
    return grid


def get_educational_prompt_suggestions(topic: str) -> List[str]:
    """
    Genera sugerencias de prompts educativos basados en un tema.
    
    Args:
        topic: Tema educativo
    
    Returns:
        Lista de sugerencias de prompts
    """
    suggestions = {
        "biologia": [
            f"Diagrama detallado de {topic} con partes etiquetadas",
            f"Ciclo de vida de {topic} con flechas y etapas",
            f"Anatomía de {topic} en estilo científico educativo",
        ],
        "quimica": [
            f"Modelo molecular de {topic} en 3D con enlaces",
            f"Reacción química de {topic} con ecuación balanceada",
            f"Tabla periódica destacando {topic}",
        ],
        "matematicas": [
            f"Representación visual de {topic} con ejemplos",
            f"Gráfica explicativa de {topic} con ejes etiquetados",
            f"Diagrama de {topic} paso a paso",
        ],
        "general": [
            f"Infografía educativa sobre {topic} con iconos",
            f"Mapa conceptual de {topic} con conexiones",
            f"Ilustración explicativa de {topic}",
        ]
    }
    
    # Detectar categoría
    topic_lower = topic.lower()
    if any(word in topic_lower for word in ["célula", "animal", "planta", "cuerpo", "órgano"]):
        category = "biologia"
    elif any(word in topic_lower for word in ["molécula", "elemento", "reacción", "átomo"]):
        category = "quimica"
    elif any(word in topic_lower for word in ["número", "ecuación", "geometría", "función"]):
        category = "matematicas"
    else:
        category = "general"
    
    return suggestions.get(category, suggestions["general"])


# Constantes útiles
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.webp']
MAX_IMAGE_DIMENSION = 2048
DEFAULT_GENERATION_PARAMS = {
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024
}






