import sounddevice as sd
import numpy as np
import joblib
import time

# Parámetros
DURACION_VENTANA = 1.0  # segundos
FS = 44100
UMBRAL_SILENCIO = 0.01
MODELO_PATH = "modelo_sonidos_rf.pkl"
# MODELO_PATH = "modelo_sonidos_knn.pkl"


# Cargar modelo entrenado
modelo = joblib.load(MODELO_PATH)
print("✅ Modelo cargado. Escuchando...\n")

def calcular_espectro(audio):
    ventana = np.hanning(len(audio))
    audio_vent = audio * ventana
    fft = np.fft.rfft(audio_vent)
    magnitudes = np.abs(fft)
    return magnitudes / np.max(magnitudes)

sonido_anterior = None

try:
    while True:
        audio = sd.rec(int(DURACION_VENTANA * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        energia_rms = np.sqrt(np.mean(audio ** 2))
        if energia_rms < UMBRAL_SILENCIO:
            prediccion = "silencio"
        else:
            espectro = calcular_espectro(audio)
            if len(espectro) > modelo.n_features_in_:
                espectro = espectro[:modelo.n_features_in_]
            elif len(espectro) < modelo.n_features_in_:
                espectro = np.pad(espectro, (0, modelo.n_features_in_ - len(espectro)))
            prediccion = modelo.predict([espectro])[0]

        if prediccion != sonido_anterior:
            print(f"\n🔊 Sonido detectado: **{prediccion.upper()}**")
            sonido_anterior = prediccion

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Monitoreo detenido por el usuario.")
