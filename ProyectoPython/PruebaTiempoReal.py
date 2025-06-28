import sounddevice as sd
import numpy as np
import time

# Parámetros
DURACION_VENTANA = 0.5  # duración de cada escucha (segundos)
FS = 44100              # frecuencia de muestreo
UMBRAL = 0.02           # umbral de energía para detectar sonido

print("Monitoreando sonidos importantes (presiona Ctrl+C para salir)...")

try:
    while True:
        # Captura de un fragmento corto
        audio = sd.rec(int(DURACION_VENTANA * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        # Calcular energía
        energia_rms = np.sqrt(np.mean(audio**2))

        # Mostrar resultado
        if energia_rms > UMBRAL:
            print("¡Sonido importante detectado!")
        else:
            print("... silencio o sonido bajo")

        time.sleep(0.1)  # espera pequeña para evitar saturar

except KeyboardInterrupt:
    print("\n🛑 Monitoreo terminado.")
