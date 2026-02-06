# Standard library
import os
import re
import shutil
import subprocess
import tempfile
from typing import List, Tuple, Union

# Third-party library (módulo de Python para ffmpeg)
import ffmpeg
import yaml

def load_config(config_path="config.yaml") -> dict:
    """
    Carga la configuración desde un archivo YAML.
    Busca el archivo 'config_path' en el mismo directorio que este script (utils.py).
    """
    # Obtiene el directorio donde está utils.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check in utils dir
    full_path = os.path.join(base_dir, config_path)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
            
    # Check in parent dir (project root)
    parent_path = os.path.join(base_dir, "..", config_path)
    if os.path.exists(parent_path):
        with open(parent_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(f"Config file not found at: {full_path} or {parent_path}")
        
    # Removed the previous single open/return



# Helpers internos (necesarios)

def _which_or_raise(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise FileNotFoundError(
            f"No se encontró '{name}' en el PATH. "
            f"Instalalo o reabrí la terminal. name={name}"
        )
    return exe


def _ffprobe_duration(input_path: str) -> float:
    ffprobe = _which_or_raise("ffprobe")
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", input_path]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def _merge_segments(segments: List[Tuple[float, float]], min_gap: float = 1e-3) -> List[Tuple[float, float]]:
    """Une segmentos solapados o contiguos (tolerancia min_gap)."""
    if not segments:
        return segments
    segments = sorted(segments, key=lambda x: x[0])
    merged = [segments[0]]
    for s, e in segments[1:]:
        ls, le = merged[-1]
        if s <= le + min_gap:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


# Funciones en uso

def crear_carpetas(base_path: str, carpetas: list) -> None:
    """Crea subcarpetas dentro de 'base_path'. Si ya existen, no hace nada."""
    for carpeta in carpetas:
        ruta_completa = os.path.join(base_path, carpeta)
        os.makedirs(ruta_completa, exist_ok=True)


def crear_carpeta_y_obsidian(ps_nn: str, vault_path: str, root_folder: str = "") -> str:
    """
    Crea las carpetas ps_nn y ps_<n+1> dentro del vault, y un .md vacío en cada una si no existe.
    Retorna la ruta absoluta de ps_nn.
    """
    base_dir = os.path.join(vault_path, root_folder) if root_folder else vault_path
    os.makedirs(base_dir, exist_ok=True)

    current_folder = os.path.join(base_dir, ps_nn)
    os.makedirs(current_folder, exist_ok=True)

    prefijo, num_str = ps_nn.split("_", 1)
    n = int(num_str)
    next_name = f"{prefijo}_{n+1}"
    next_folder = os.path.join(base_dir, next_name)
    os.makedirs(next_folder, exist_ok=True)

    def _crear_md(folder: str, numero: int, nombre: str):
        filename = f"Pizza stream {numero} ({nombre}).md"
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            open(path, "w", encoding="utf-8").close()
            print("Archivo vacío creado:", path)
        else:
            print("Ya existía:", path)

    _crear_md(current_folder, n, ps_nn)
    _crear_md(next_folder, n + 1, next_name)

    return current_folder


def sincronizar_audio_video(input_path: str, output_path: str, desfase_ms: int) -> str:
    """
    Sincroniza audio y video desplazando uno de los streams según 'desfase_ms'.
    >0 retrasa audio; <0 retrasa video. Devuelve la ruta de salida o None si falla.
    """
    desfase_s = desfase_ms / 1000.0

    if desfase_ms >= 0:
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-itsoffset", str(desfase_s),
            "-i", input_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c", "copy",
            "-y",
            output_path
        ]
    else:
        desfase_s_abs = abs(desfase_s)
        cmd = [
            "ffmpeg",
            "-itsoffset", str(desfase_s_abs),
            "-i", input_path,
            "-i", input_path,
            "-map", "1:v",
            "-map", "0:a",
            "-c", "copy",
            "-y",
            output_path
        ]

    print("Ejecutando comando de sincronización:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Error al sincronizar audio/video:")
        print(result.stderr)
        return None

    print(f"Sincronización completada. Archivo guardado en: {output_path}")
    return output_path


def remove_silence_precise(
    input_path: str,
    output_path: str,
    silence_threshold: float = -35.0,
    silence_min_duration: float = 0.5,
    buffer: float = 0.2,
    buffer_start: float | None = None,
    buffer_end: float | None = None,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    preset: str = "medium",
    use_fallback_silenceremove: bool = True,
    workers: int | None = None,
    hwaccel: str | None = None
) -> str:
    """
    Recorta silencios con precisión (detección con silencedetect + trim/atrim + concat).
    Soporta procesamiento en paralelo dividiendo el trabajo en chunks.
    
    - workers: número de procesos paralelos.
      Si workers es None o < 1, usa todos los cores disponibles (os.cpu_count()).
    """
    import math
    import concurrent.futures

    # Si no se especifican buffers específicos, usar el buffer genérico
    if buffer_start is None:
        buffer_start = buffer
    if buffer_end is None:
        buffer_end = buffer

    ffmpeg_bin = _which_or_raise("ffmpeg")
    _which_or_raise("ffprobe")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input no existe: {input_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 1) Detección de silencios (igual que antes)
    cmd_detection = [
        ffmpeg_bin, "-hide_banner", "-vn",
        "-i", input_path,
        "-af", f"silencedetect=n={silence_threshold}dB:d={silence_min_duration}",
        "-f", "null", "-"
    ]
    try:
        proc = subprocess.run(cmd_detection, capture_output=True, text=True, check=True)
        log = proc.stderr
    except subprocess.CalledProcessError as e:
        log = e.stderr or e.stdout or ""

    # 2) Parseo de silencios
    silence_periods: List[Tuple[float, float]] = []
    last_start = None
    for line in (log or "").splitlines():
        if "silence_start:" in line:
            m = re.search(r"silence_start:\s*([0-9.]+)", line)
            last_start = float(m.group(1)) if m else None
        elif "silence_end:" in line:
            m_end = re.search(r"silence_end:\s*([0-9.]+)", line)
            m_dur = re.search(r"silence_duration:\s*([0-9.]+)", line)
            if m_end:
                end = float(m_end.group(1))
                if last_start is not None:
                    start = float(last_start)
                elif m_dur:
                    start = end - float(m_dur.group(1))
                else:
                    continue
                if end > start:
                    silence_periods.append((start, end))
                last_start = None

    silence_periods.sort(key=lambda x: x[0])

    # 3) Duración total
    try:
        total_duration = _ffprobe_duration(input_path)
    except Exception:
        total_duration = silence_periods[-1][1] if silence_periods else 0.0

    # 4) Segmentos con audio
    segments: List[Tuple[float, float]] = []
    prev_end = 0.0
    for s_sil, e_sil in silence_periods:
        if s_sil > prev_end:
            s = max(0.0, prev_end - buffer_start)
            e = min(total_duration, s_sil + buffer_end)
            if e > s:
                segments.append((s, e))
        prev_end = max(prev_end, e_sil)

    if prev_end < total_duration:
        s = max(0.0, prev_end - buffer_start)
        e = total_duration
        if e > s:
            segments.append((s, e))

    segments = _merge_segments(segments)

    if not segments:
        print("No se encontraron silencios o segmentos válidos. Copiando original.")
        subprocess.run([ffmpeg_bin, "-y", "-i", input_path, "-c", "copy", output_path], check=True)
        return output_path

    # Función interna para procesar un chunk
    def _process_chunk_segments(chunk_segs, chunk_idx, tmp_dir):
        chunk_out = os.path.join(tmp_dir, f"chunk_{chunk_idx:04d}.mp4")
        
        filter_lines = []
        for i, (s, e) in enumerate(chunk_segs):
            filter_lines.append(
                f"[0:v]trim=start={s:.6f}:end={e:.6f},setpts=PTS-STARTPTS[v{i}];"
                f"[0:a]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{i}];"
            )
        concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(chunk_segs)))
        filter_complex = "\n".join(filter_lines) + f"\n{concat_inputs}concat=n={len(chunk_segs)}:v=1:a=1[v][a]"

        # Usar utf-8 para el archivo de filtro
        filter_file = os.path.join(tmp_dir, f"filter_{chunk_idx:04d}.txt")
        with open(filter_file, "w", encoding="utf-8") as f:
            f.write(filter_complex)

        cmd = [
            ffmpeg_bin, "-y", "-hide_banner"
        ]
        
        if hwaccel:
            cmd.extend(["-hwaccel", hwaccel])
            
        cmd.extend([
            "-i", input_path,
            "-filter_complex_script", filter_file,
            "-map", "[v]", "-map", "[a]",
            "-c:v", video_codec, "-preset", preset,
            "-c:a", audio_codec,
            chunk_out
        ])
        
        # Ejecutar ffmpeg
        # print(f"Procesando chunk {chunk_idx} ({len(chunk_segs)} segmentos)...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"Error en chunk {chunk_idx}: {res.stderr[-2000:]}")
            raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)
        
        return chunk_out

    # 5) Procesamiento (puede ser paralelo)
    if workers is None or workers < 1:
        workers = os.cpu_count() or 1
        
    current_workers = max(1, min(workers, len(segments)))
    
    # Crear directorio temporal para los chunks
    with tempfile.TemporaryDirectory() as temp_dir:
        # Dividir segmentos en chunks
        chunk_size = math.ceil(len(segments) / current_workers)
        chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
        
        print(f"Dividiendo {len(segments)} segmentos en {len(chunks)} tareas (workers={current_workers})...")

        temp_files_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=current_workers) as executor:
            future_to_idx = {
                executor.submit(_process_chunk_segments, chunk, idx, temp_dir): idx 
                for idx, chunk in enumerate(chunks)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    out_file = future.result()
                    temp_files_map[idx] = out_file
                    print(f"Chunk {idx} completado.")
                except Exception as exc:
                    print(f"Chunk {idx} generó una excepción: {exc}")
                    # Opcional: cancelar otros futures?
                    raise exc

        # Ordenar archivos resultantes
        ordered_files = [temp_files_map[i] for i in range(len(chunks))]

        # Concatenar resultados
        if len(ordered_files) == 1:
            # Si solo hay uno, moverlo al destino final
            shutil.move(ordered_files[0], output_path)
            print(f"Proceso completado. Archivo generado en: {output_path}")
        else:
            # Generar lista para concat demuxer
            concat_list = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list, "w", encoding="utf-8") as f:
                for tf in ordered_files:
                    # Windows path compatibility
                    safe_path = tf.replace("\\", "/")
                    f.write(f"file '{safe_path}'\n")

            print("Concatenando chunks finales...")
            cmd_merge = [
                ffmpeg_bin, "-y", "-hide_banner",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                output_path
            ]
            subprocess.run(cmd_merge, check=True)
            print(f"Proceso completado (merged). Archivo generado en: {output_path}")

    return output_path



def text_input_clips(texto: str) -> list:
    """
    Convierte texto de líneas en timeline de tuplas para clips().
    Acepta tiempos 'm.ss' o 'mm.ss' y los normaliza a 'mm:ss'.
    """
    def normalizar_tiempo(tiempo_str: str) -> str:
        if ':' in tiempo_str:
            return tiempo_str
        if '.' in tiempo_str:
            minutos, segundos = tiempo_str.split('.', 1)
            return f"{int(minutos):02d}:{int(segundos):02d}"
        return tiempo_str

    timeline = []
    for linea in texto.strip().splitlines():
        linea = linea.strip()
        if not linea:
            continue
        tokens = linea.split()
        tiempo = normalizar_tiempo(tokens[0])

        if len(tokens) == 1:
            timeline.append((tiempo,))
            continue

        try:
            duracion = int(tokens[1])
            if len(tokens) > 2:
                stream = " ".join(tokens[2:])
                timeline.append((tiempo, duracion, stream))
            else:
                timeline.append((tiempo, duracion))
        except ValueError:
            stream = " ".join(tokens[1:])
            timeline.append((tiempo, 60, stream))

    return timeline


def clips(input_path: str, timeline: list, output_path: str) -> list:
    """
    Recorta varios clips de un video, dado un timeline de (inicio, [duración], [stream_name]).
    Requiere ffmpeg-python.
    """
    os.makedirs(output_path, exist_ok=True)

    def time_to_seconds(t_str: str) -> int:
        partes = list(map(int, t_str.split(":")))
        if len(partes) == 1:
            return partes[0]
        if len(partes) == 2:
            return partes[0] * 60 + partes[1]
        if len(partes) == 3:
            return partes[0] * 3600 + partes[1] * 60 + partes[2]
        raise ValueError(f"Formato de tiempo no válido: {t_str}")

    output_files = []
    for idx, item in enumerate(timeline):
        if not isinstance(item, (list, tuple)) or not (1 <= len(item) <= 3):
            raise ValueError("Cada elemento de timeline debe tener 1 a 3 valores.")

        tiempo_str = item[0]
        duracion = item[1] if len(item) >= 2 and item[1] is not None else 60
        stream_name = item[2] if len(item) == 3 else None

        start_sec = time_to_seconds(tiempo_str)
        timestamp_sanitized = tiempo_str.replace(":", "m") + "s"

        clip_base = stream_name if stream_name else timestamp_sanitized
        id_str = f"{idx:02d}"
        cut_filename = f"{id_str}_{clip_base}_{timestamp_sanitized}_{duracion}s.mp4"
        cut_filepath = os.path.join(output_path, cut_filename)

        try:
            (
                ffmpeg
                .input(input_path, ss=start_sec, t=duracion)
                .output(
                    cut_filepath,
                    vcodec='libx264', acodec='aac', preset='medium',
                    profile='high', pix_fmt='yuv420p', ac=2,
                    **{'b:v': '2500k', 'b:a': '128k', 'movflags': '+faststart'}
                )
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
            print(f"Corte {id_str}: {cut_filepath}")
            output_files.append(cut_filepath)
        except ffmpeg.Error as e:
            print(f"Error recortando clip {id_str}: {e.stderr.decode()}")

    return output_files


def convert_videos_vertical(
    input_paths: Union[str, List[str]],
    output_dir: str,
    cut_pixel: int,
    resolution: str = "720p",
    bar_height: int = 10
) -> List[str]:
    """
    Convierte video(s) 16:9 a formato vertical apilando izquierda (arriba) + barra negra + derecha (abajo).
    - resolution: '720p' -> (1280x720) o '1080p' -> (1920x1080)
    - cut_pixel: punto de corte horizontal en la resolución original
    """
    paths: List[str] = [input_paths] if isinstance(input_paths, str) else (input_paths or [])
    if not paths:
        raise ValueError("Debes proporcionar al menos un video de entrada.")
    os.makedirs(output_dir, exist_ok=True)

    if resolution.lower() == "1080p":
        iw, ih = 1920, 1080
    else:
        iw, ih = 1280, 720

    if not (0 < cut_pixel < iw):
        raise ValueError(f"cut_pixel debe estar entre 1 y {iw - 1} para resolution={resolution}")

    left_w = cut_pixel
    right_w = iw - cut_pixel

    outputs: List[str] = []
    for in_path in paths:
        if not os.path.isfile(in_path):
            print(f"⚠️  Archivo no encontrado, se omite: {in_path}")
            continue

        left_h = round(720 * ih / left_w)
        right_h = round(720 * ih / right_w)

        if left_h >= 640:
            oy = (left_h - 640) // 2
            left_chain = (
                f"crop=w={left_w}:h={ih}:x=0:y=0,"
                "scale=720:-1,"
                f"crop=720:640:0:{oy}"
            )
        else:
            left_chain = (
                f"crop=w={left_w}:h={ih}:x=0:y=0,"
                "scale=-1:640,"
                "pad=720:640:(720-iw)/2:0"
            )

        if right_h >= 640:
            oy = (right_h - 640) // 2
            right_chain = (
                f"crop=w={right_w}:h={ih}:x={left_w}:y=0,"
                "scale=720:-1,"
                f"crop=720:640:0:{oy}"
            )
        else:
            right_chain = (
                f"crop=w={right_w}:h={ih}:x={left_w}:y=0,"
                "scale=-1:640,"
                "pad=720:640:(720-iw)/2:0"
            )

        filter_complex = (
            "[0:v]split=2[left][right];"
            f"[left]{left_chain}[lout];"
            f"[right]{right_chain}[rout];"
            f"[lout]pad=720:{640 + bar_height}:0:0:color=black[lout_pad];"
            "[lout_pad][rout]vstack=2[outv]"
        )

        base = os.path.splitext(os.path.basename(in_path))[0]
        out_name = f"{base}_vertical.mp4"
        out_path = os.path.join(output_dir, out_name)

        cmd = [
            "ffmpeg",
            "-i", in_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-c:a", "copy",
            "-y",
            out_path
        ]
        print("Ejecutando:", " ".join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            print(f"Error procesando {in_path}:\n{result.stderr.decode()}")
        else:
            print(f"Generado → {out_path}")
            outputs.append(out_path)

    return outputs
