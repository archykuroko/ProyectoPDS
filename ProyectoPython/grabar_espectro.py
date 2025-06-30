import sounddevice as sd
import numpy as np
import os
import json

# --- CONFIGURACIÓN ---
DURACION = 3.0  # segundos de grabación
FS = 44100      # Hz
CARPETA_SONIDOS = "base_sonidos"

# Crear carpeta si no existe
os.makedirs(CARPETA_SONIDOS, exist_ok=True)

# --- ETIQUETA ---
etiqueta = input("Ingresa la etiqueta del sonido (ej: 'alarma', 'bebe', 'timbre'): ").strip().lower()

print(f"Grabando sonido para la etiqueta '{etiqueta}'...")
audio = sd.rec(int(DURACION * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()

# Aplicar ventana para suavizar bordes
ventana = np.hanning(len(audio))
audio_ventaneado = audio * ventana

# Calcular FFT (espectro)
fft = np.fft.rfft(audio_ventaneado)
frecuencias = np.fft.rfftfreq(len(audio_ventaneado), 1 / FS)
magnitudes = np.abs(fft)

# Normalizar espectro
magnitudes_norm = magnitudes / np.max(magnitudes)

# Guardar como archivo JSON
datos = {
    "etiqueta": etiqueta,
    "frecuencias": frecuencias.tolist(),
    "magnitudes": magnitudes_norm.tolist()
}
ruta_archivo = os.path.join(CARPETA_SONIDOS, f"{etiqueta}.json")
with open(ruta_archivo, "w") as f:
    json.dump(datos, f)

print(f"Espectro del sonido '{etiqueta}' guardado en {ruta_archivo}")
