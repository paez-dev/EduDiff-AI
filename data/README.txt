═══════════════════════════════════════════════════════════════════════════════
                        EduDiff XL - Directorio de Datos
═══════════════════════════════════════════════════════════════════════════════

Este directorio está destinado a almacenar datos de ejemplo y recursos 
adicionales para el sistema EduDiff.

ESTRUCTURA SUGERIDA:
───────────────────────────────────────────────────────────────────────────────

data/
├── README.txt           # Este archivo
├── examples/            # Imágenes de ejemplo para ControlNet
│   ├── sketches/       # Bocetos de entrada
│   ├── diagrams/       # Diagramas base
│   └── references/     # Imágenes de referencia
├── prompts/            # Colecciones de prompts
│   ├── biology.txt     # Prompts de biología
│   ├── chemistry.txt   # Prompts de química
│   ├── math.txt        # Prompts de matemáticas
│   └── general.txt     # Prompts generales
└── templates/          # Plantillas de composición
    └── layouts/        # Layouts predefinidos

NOTAS:
───────────────────────────────────────────────────────────────────────────────

1. Los datos de ejemplo no están incluidos en el repositorio por tamaño.
2. Para usar ControlNet, coloca imágenes de entrada en examples/.
3. Las imágenes deben estar en formato PNG o JPG.
4. Resolución recomendada: 1024x1024 píxeles.

FORMATOS SOPORTADOS:
───────────────────────────────────────────────────────────────────────────────

- Imágenes: PNG, JPG, JPEG, WEBP
- Prompts: TXT (UTF-8)
- Metadatos: JSON

EJEMPLO DE USO:
───────────────────────────────────────────────────────────────────────────────

Para usar una imagen como guía con ControlNet:

1. Coloca tu imagen en data/examples/
2. En la aplicación, sube la imagen en "Imagen Guía"
3. Selecciona el tipo de ControlNet apropiado:
   - Sketch: para bocetos a mano
   - Canny: para imágenes con bordes definidos
   - Depth: para mapas de profundidad
   - Lineart: para arte lineal limpio

═══════════════════════════════════════════════════════════════════════════════

