# Pizza Stream Pipeline 🎬🍕

Pipeline para automatizar el flujo del Pizza Stream:
- sincronizar audio/video,
- recortar silencios,
- generar clips horizontales,
- convertirlos a vertical,
- crear estructura de carpetas/notas en Obsidian,
- estilizar el chat de YouTube en OBS.

## Archivos
- **pizza_stream_pipeline.ipynb**: ejecuta todo el proceso para un stream `ps_nn`.
- **utils.py**: funciones para ffmpeg (sincronizar, cortar silencios, clips, vertical), armado de carpetas y parsing de timelines.
- **OBS_chat.css**: estilo transparente, limpio y animado para el chat en OBS.

## Requisitos
- Python 3.12+
- ffmpeg + ffprobe en el PATH
- `pip install ffmpeg-python`

## Uso
1. Configurar `ps_nn` y rutas en el notebook.  
2. Ejecutar: crear carpetas → sincronizar → recortar silencios.  
3. Editar el video manualmente.  
4. Definir lista de clips en texto → generar clips horizontales.  
5. Convertir a vertical.  
6. En OBS, pegar `OBS_chat.css` como Custom CSS.

