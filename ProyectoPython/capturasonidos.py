import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
DURACION = 1.0  # segundos
FS = 44100      # Hz

# Capturar audio
print("Grabando audio para analizar espectro...")
audio = sd.rec(int(DURACION * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()

# Aplicar ventana
ventana = np.hanning(len(audio))
audio_ventaneado = audio * ventana

# Calcular FFT
fft = np.fft.rfft(audio_ventaneado)
frecuencias = np.fft.rfftfreq(len(audio_ventaneado), 1/FS)
magnitudes = np.abs(fft)

# Mostrar gráfico
plt.figure(figsize=(10, 5))
plt.plot(frecuencias, magnitudes)
plt.title("Espectro de Frecuencia del Sonido")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.xlim(0, 5000)  # recortar a lo audible útil
plt.show()
