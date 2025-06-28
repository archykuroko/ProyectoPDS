import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

#Parámetros
DURACION = 3  # segundos
FS = 44100    # Hz (frecuencia de muestreo)
UMBRAL = 0.02  # energía mínima para considerar "sonido importante"

# Captura de audio
print("Grabando sonido por 3 segundos...")
audio = sd.rec(int(DURACION * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()
print("Grabación completada.")

#Procesamiento
audio = audio.flatten()  # convierte a 1D
energia_rms = np.sqrt(np.mean(audio**2))  # raíz cuadrada de la media cuadrática

#Resultado
print(f"Energía RMS: {energia_rms:.4f}")
if energia_rms > UMBRAL:
    print("¡Sonido importante detectado!")
else:
    print("Sonido bajo o silencio.")

#Visualización
plt.plot(audio)
plt.title("Forma de onda capturada")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
