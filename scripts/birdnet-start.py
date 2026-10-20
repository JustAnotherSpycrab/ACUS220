import os
import glob
import time
import wave
import contextlib
import pandas as pd

# ==============================================================================
# 1. CONTROL DE RECURSOS DE HARDWARE (CPU THREAD LIMITING)
# ==============================================================================
# Limitar librerías numéricas a 4 hilos para evitar que el procesador sature al 100%
NUM_THREADS = "4"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS

import onnxruntime as ort
from birdnet import BirdNETV3

# ==============================================================================
# 2. DEFINICIÓN DE RUTAS Y PARÁMETROS GLOBALES
# ==============================================================================
BASE_DIR = r"C:\Users\punto\Documents\ACUS220"
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
AUDIT_LOG_CSV = os.path.join(OUTPUT_DIR, "audit_log.csv")

CONFIDENCE_THRESHOLD = 0.30  # Descarta detecciones ruidosas (<30%) para aligerar I/O
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Coordenadas geográficas por defecto (pueden ser None si se procesa a nivel global)
# Ejemplo: Valdivia, Chile (Lat: -39.81, Lon: -73.24)
DEFAULT_LAT = -39.8142
DEFAULT_LON = -73.2459

# ==============================================================================
# 3. MÓDULO DE AUDITORÍA TÉCNICA Y PRE-VALIDACIÓN ACÚSTICA
# ==============================================================================
def auditar_audio(audio_path):
    """
    Inspecciona metadatos del archivo de audio antes de la inferencia.
    Verifica que el archivo no esté corrupto, revisa duración y canales.
    """
    audit_data = {
        "filename": os.path.basename(audio_path),
        "channels": None,
        "sample_rate": None,
        "duration_sec": 0.0,
        "valid_for_inference": False,
        "observation": "OK"
    }
    
    try:
        with contextlib.closing(wave.open(audio_path, 'rb')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            channels = f.getnchannels()
            duration = frames / float(rate)
            
            audit_data["channels"] = channels
            audit_data["sample_rate"] = rate
            audit_data["duration_sec"] = round(duration, 2)
            
            # Criterios de auditoría: audio mínimo de 3 segundos
            if duration < 3.0:
                audit_data["observation"] = "Rechazado: Duración menor a 3 s."
            else:
                audit_data["valid_for_inference"] = True
                if channels > 1:
                    audit_data["observation"] = f"Aviso: Audio multicanal ({channels}ch), se mezclará a mono."
    except Exception as e:
        audit_data["observation"] = f"Error de lectura: {str(e)}"
        
    return audit_data

# ==============================================================================
# 4. INICIALIZACIÓN DEL MODELO CON ACELERACIÓN GPU
# ==============================================================================
print("=" * 70)
print("INICIALIZANDO PIPELINE BIOACÚSTICO (ONNX RUNTIME GPU + AUDITORÍA)")
print("=" * 70)

# Configuración de ejecución en ONNX Runtime
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.inter_op_num_threads = 2

# Instanciar BirdNET V3 con proveedores de ejecución priorizados
model = BirdNETV3(
    execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)


# 5. EJECUCIÓN POR LOTES
audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
print(f"Archivos encontrados en {AUDIO_DIR}: {len(audio_files)}\n")

audit_records = []
total_start_time = time.time()

for idx, audio_path in enumerate(audio_files, start=1):
    filename = os.path.basename(audio_path)
    csv_name = os.path.splitext(filename)[0] + ".csv"
    output_csv = os.path.join(OUTPUT_DIR, csv_name)
    
    print(f"[{idx}/{len(audio_files)}] Auditando: {filename}...")
    
    # 1. Auditoría previa
    audit = auditar_audio(audio_path)
    audit_records.append(audit)
    
    if not audit["valid_for_inference"]:
        print(f"  -> Omitido: {audit['observation']}")
        continue
        
    print(f"  -> Audio válido ({audit['duration_sec']}s, {audit['sample_rate']}Hz, {audit['channels']}ch).")
    print(f"  -> Ejecutando inferencia en GPU...")
    
    t_inference_start = time.time()
    
    # 2. Inferencia BirdNET V3
    try:
        # Se intenta pasar el umbral si la API lo admite directamente
        predictions = model.predict(
            audio_path,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            latitude=DEFAULT_LAT,
            longitude=DEFAULT_LON
        )
    except TypeError:
        # Fallback estándar si los argumentos varían en la versión instalada
        predictions = model.predict(audio_path)

    # 3. Filtrado y exportación estructurada
    if hasattr(predictions, 'to_csv'):
        # Si devuelve DataFrame, filtramos por umbral antes de escribir
        if hasattr(predictions, 'confidence'):
            predictions = predictions[predictions['confidence'] >= CONFIDENCE_THRESHOLD]
        predictions.to_csv(output_csv, index=False)
    else:
        predictions.to_csv(output_csv)
        
    t_elapsed = time.time() - t_inference_start
    print(f"  -> Exportado en {t_elapsed:.2f} s: {csv_name}")

# 6. reporte
audit_df = pd.DataFrame(audit_records)
audit_df.to_csv(AUDIT_LOG_CSV, index=False)

total_elapsed = time.time() - total_start_time
print("\n" + "=" * 70)
print(f"PROCESO FINALIZADO CON ÉXITO")
print(f"Tiempo total: {total_elapsed:.2f} segundos")
print(f"Reporte de auditoría generado en: {AUDIT_LOG_CSV}")
print("=" * 70)
