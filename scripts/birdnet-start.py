import os
import sys
import glob
import time
import wave
import warnings
import contextlib
import concurrent.futures

# ==============================================================================
# 1. LÍMITE DE HILOS (debe ir ANTES de importar numpy / onnxruntime / birdnet)
# ==============================================================================
MAX_HILOS = 4
for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[var] = str(MAX_HILOS)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")



def _registrar_dlls_cuda():
    """En Windows, Python no usa el PATH para cargar DLL de dependencias.
    Registra las carpetas de CUDA/cuDNN (incluida bin\\x64 de CUDA 13) y las
    antepone al PATH para que también las hereden los procesos de trabajo."""
    if os.name != "nt":
        return
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    candidatos = []
    for k, v in os.environ.items():
        if k.startswith("CUDA_PATH") and v:
            candidatos += [os.path.join(v, "bin"), os.path.join(v, "bin", "x64")]
    cuda_root = os.path.join(pf, "NVIDIA GPU Computing Toolkit", "CUDA", "v*")
    cudnn_root = os.path.join(pf, "NVIDIA", "CUDNN", "v*", "bin")
    candidatos += glob.glob(os.path.join(cuda_root, "bin"))
    candidatos += glob.glob(os.path.join(cuda_root, "bin", "x64"))
    candidatos += glob.glob(cudnn_root)
    candidatos += glob.glob(os.path.join(cudnn_root, "*"))

    vistos = []
    for d in candidatos:
        if os.path.isdir(d) and d not in vistos:
            vistos.append(d)
            with contextlib.suppress(Exception):
                os.add_dll_directory(d)
    if vistos:
        os.environ["PATH"] = os.pathsep.join(vistos) + os.pathsep + os.environ.get("PATH", "")


_registrar_dlls_cuda()

import onnxruntime as ort

ort.set_default_logger_severity(4)  # solo errores fatales

# Precarga las DLL de CUDA/cuDNN si están instaladas (onnxruntime-gpu >= 1.21)
if hasattr(ort, "preload_dlls"):
    with contextlib.suppress(Exception):
        ort.preload_dlls()

# Límite de hilos en cada sesión ONNX (sin tocar los proveedores: la GPU se
# elige con el parámetro `device` de model.predict()).
_init_original = ort.InferenceSession.__init__


def _init_limitado(self, path_or_bytes, sess_options=None, providers=None,
                   provider_options=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = MAX_HILOS
    sess_options.inter_op_num_threads = 1
    sess_options.log_severity_level = 4
    _init_original(self, path_or_bytes, sess_options=sess_options,
                   providers=providers, provider_options=provider_options, **kwargs)


ort.InferenceSession.__init__ = _init_limitado

import birdnet  # noqa: E402

# ==============================================================================
# 2. RUTAS Y CONFIGURACIÓN
# ==============================================================================
BASE_DIR = r"C:\Users\punto\Documents\ACUS220"
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CONFIDENCE_THRESHOLD = 0.40   # se aplica con default_confidence_threshold
ARCHIVOS_EN_PARALELO = 2      # hilos de ThreadPoolExecutor
DEVICE = "GPU"                # "GPU", "GPU:0" o "CPU"
BATCH_SIZE = 16               # segmentos de 3 s por lote en la GPU
DURACION_MINIMA_S = 3.0

OMITIR_PROCESADOS = True      # si ya existe el CSV de un audio, se salta (permite reanudar)
ARCHIVO_PARQUET = os.path.join(BASE_DIR, "resumen_detecciones.parquet")  # archivo madre

model = None


def cargar_modelo():
    global model
    if DEVICE.upper().startswith("GPU") and \
            "CUDAExecutionProvider" not in ort.get_available_providers():
        sys.exit(
            "❌ CUDAExecutionProvider no está disponible en onnxruntime.\n"
            "   Instala 'onnxruntime-gpu' (sin 'onnxruntime') y las DLL de CUDA/cuDNN."
        )
    model = birdnet.load("acoustic", "3.0", "onnx")
    print(f"✅ Modelo cargado. Dispositivo solicitado: {DEVICE}")


# ==============================================================================
# 3. PROCESAMIENTO DE CADA ARCHIVO
# ==============================================================================
def guardar_csv(predictions, output_csv):
    """Escribe en un archivo temporal y lo renombra al terminar, para que un
    corte a mitad de escritura no deje un CSV incompleto marcado como 'listo'."""
    tmp_csv = output_csv[:-4] + ".tmp.csv"
    if hasattr(predictions, "to_csv"):
        predictions.to_csv(tmp_csv)
    else:
        import pandas as pd
        pd.DataFrame(predictions).to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, output_csv)


def procesar_un_audio(audio_path):
    """Devuelve (estado, mensaje) con estado en {'ok', 'skip', 'error'}."""
    filename = os.path.basename(audio_path)
    output_csv = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".csv")
    t_inicio = time.time()

    if OMITIR_PROCESADOS and os.path.exists(output_csv):
        return "skip", f"[{filename}] Ya procesado."

    # Auditoría rápida de duración (solo WAV PCM; si falla, decide BirdNET)
    try:
        with contextlib.closing(wave.open(audio_path, "rb")) as f:
            if f.getnframes() / float(f.getframerate()) < DURACION_MINIMA_S:
                return "skip", f"[{filename}] Omitido: dura menos de {DURACION_MINIMA_S:.0f} s."
    except wave.Error:
        pass
    except Exception as e:
        return "error", f"[{filename}] Error leyendo archivo: {e}"

    try:
        predictions = model.predict(
            audio_path,
            top_k=None,                                   # todas las especies sobre el umbral
            default_confidence_threshold=CONFIDENCE_THRESHOLD,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            n_producers=1,
            n_workers=1,
            show_stats=None,                              # sin barras ni logs
        )
        guardar_csv(predictions, output_csv)
        return "ok", f"[{filename}] OK en {time.time() - t_inicio:.2f} s"
    except Exception as e:
        return "error", f"[{filename}] Error durante la inferencia: {e}"


def consolidar_parquet():
    """Une todos los CSV de OUTPUT_DIR en un único .parquet (los CSV se conservan)."""
    import pandas as pd

    csvs = sorted(
        p for p in glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
        if not p.endswith(".tmp.csv")
    )
    partes = []
    for ruta in csvs:
        try:
            df = pd.read_csv(ruta)
        except pd.errors.EmptyDataError:
            continue  # audio sin detecciones: CSV vacío
        if df.empty:
            continue
        # Elimina columna de índice residual si to_csv la escribió
        df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")])
        df.insert(0, "archivo_audio", os.path.splitext(os.path.basename(ruta))[0])
        partes.append(df)

    if not partes:
        return None, len(csvs)

    total = pd.concat(partes, ignore_index=True)
    try:
        total.to_parquet(ARCHIVO_PARQUET, index=False)
    except Exception:
        # Columnas con tipos mixtos: se pasan a texto y se reintenta
        for col in total.select_dtypes(include="object").columns:
            total[col] = total[col].astype(str)
        total.to_parquet(ARCHIVO_PARQUET, index=False)
    return len(total), len(csvs)


# ==============================================================================
# 4. BLOQUE PRINCIPAL
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))

    if not audio_files:
        print("⚠️ No se encontraron archivos .wav en la carpeta 'audio'. ¡Revisa la ruta!")
        return

    cargar_modelo()
    t0 = time.time()
    conteo = {"ok": 0, "skip": 0, "error": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=ARCHIVOS_EN_PARALELO) as executor:
        futuros = [executor.submit(procesar_un_audio, p) for p in audio_files]
        for futuro in concurrent.futures.as_completed(futuros):
            estado, mensaje = futuro.result()
            conteo[estado] += 1
            if estado != "skip" or "Omitido" in mensaje:
                print(mensaje)  # los "ya procesado" no se imprimen para no llenar la consola

    print(f"\nProcesados: {conteo['ok']} | Ya existentes: {conteo['skip']} | Errores: {conteo['error']}")

    try:
        filas, n_csv = consolidar_parquet()
        if filas is None:
            print("ℹ️ No hay detecciones para consolidar en el .parquet.")
        else:
            print(f"📦 Parquet: {filas} detecciones de {n_csv} CSV -> {ARCHIVO_PARQUET}")
    except ImportError:
        print("⚠️ Falta 'pyarrow' para crear el .parquet: pip install pyarrow")
    except Exception as e:
        print(f"⚠️ No se pudo crear el .parquet: {e}")

    print("\n" + "=" * 70)
    print(f"¡LOTE COMPLETO! {len(audio_files)} archivos en {time.time() - t0:.2f} s")
    print(f"CSV individuales en: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    
