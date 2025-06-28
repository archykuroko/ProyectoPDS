import tkinter as tk
import sounddevice as sd
import numpy as np
import threading

# Parámetros de audio
DURACION_VENTANA = 0.5  # segundos
FS = 44100  # frecuencia de muestreo
UMBRAL = 0.02  # umbral de energía

# Crear la ventana principal
root = tk.Tk()
root.title("Detector de Sonido Importante")
root.geometry("400x300")
root.configure(bg="black")

# Etiqueta de estado
estado_label = tk.Label(root, text="Esperando sonido...", font=("Arial", 20), fg="white", bg="black")
estado_label.pack(expand=True)

# Función para actualizar la interfaz
def actualizar_estado(detectado):
    if detectado:
        root.configure(bg="red")
        estado_label.config(text="🔔 ¡Sonido Detectado!", bg="red")
    else:
        root.configure(bg="black")
        estado_label.config(text="🎧 Escuchando...", bg="black")

# Función que monitorea sonido continuamente
def monitorear_sonido():
    while True:
        audio = sd.rec(int(DURACION_VENTANA * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        energia_rms = np.sqrt(np.mean(audio**2))
        detectado = energia_rms > UMBRAL

        # Actualizar la interfaz desde el hilo principal
        root.after(0, actualizar_estado, detectado)

# Iniciar hilo de monitoreo
hilo = threading.Thread(target=monitorear_sonido, daemon=True)
hilo.start()

# Iniciar ventana
root.mainloop()
