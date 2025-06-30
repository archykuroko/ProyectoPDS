import numpy as np
import sounddevice as sd
import joblib
import json
import os
from tkinter import Tk, Label, StringVar, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuración
FS = 44100
DURACION = 1.0
UMBRAL_SILENCIO = 0.01
CARPETA_ESPECTROS = "base_sonidos"
MODELO_PATH = "modelo_sonidos_rf.pkl"
UMBRAL_COINCIDENCIA = 0.92

def calcular_espectro(audio):
    ventana = np.hanning(len(audio))
    audio_win = audio * ventana
    fft = np.fft.rfft(audio_win)
    magnitudes = np.abs(fft)
    if np.max(magnitudes) == 0:
        return magnitudes
    return magnitudes / np.max(magnitudes)

def cargar_base():
    base = {}
    for archivo in os.listdir(CARPETA_ESPECTROS):
        if archivo.endswith(".json"):
            with open(os.path.join(CARPETA_ESPECTROS, archivo), "r") as f:
                datos = json.load(f)
                base[archivo] = {
                    "etiqueta": datos["etiqueta"],
                    "magnitudes": np.array(datos["magnitudes"])
                }
    return base

def buscar_coincidencia(espectro, base):
    mejor_etiqueta = None
    mejor_score = 0
    for item in base.values():
        ref = item["magnitudes"]
        ref = ref[:len(espectro)]
        espectro_cortado = espectro[:len(ref)]
        score = np.dot(espectro_cortado, ref) / (np.linalg.norm(espectro_cortado) * np.linalg.norm(ref))
        if score > mejor_score:
            mejor_score = score
            mejor_etiqueta = item["etiqueta"]
    if mejor_score >= UMBRAL_COINCIDENCIA:
        return mejor_etiqueta, f"Coincidencia directa ({mejor_score:.2f})"
    return None, None

modelo = joblib.load(MODELO_PATH)
base = cargar_base()
num_features = modelo.n_features_in_

# GUI
ventana = Tk()
ventana.title("Detector de Sonidos con Espectro")
ventana.geometry("800x500")
ventana.resizable(False, False)
style = ttk.Style()
style.theme_use("clam")

var_sonido = StringVar(value="Esperando sonido...")
var_metodo = StringVar(value="Método: -")

Label(ventana, text="🎧 Detector de Sonidos", font=("Segoe UI", 20, "bold")).pack(pady=10)
Label(ventana, textvariable=var_sonido, font=("Segoe UI", 16), foreground="blue").pack()
Label(ventana, textvariable=var_metodo, font=("Segoe UI", 12), foreground="gray").pack(pady=5)

# Matplotlib dentro de tkinter
fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
ax.set_title("Espectro de sonido (FFT)")
ax.set_xlabel("Frecuencia (Hz)")
ax.set_ylabel("Amplitud")
linea_fft, = ax.plot([], [], color='orange')
canvas = FigureCanvasTkAgg(fig, master=ventana)
canvas.get_tk_widget().pack(pady=10)

def detectar():
    audio = sd.rec(int(DURACION * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    energia = np.sqrt(np.mean(audio ** 2))

    if energia < UMBRAL_SILENCIO:
        var_sonido.set("Silencio")
        var_metodo.set("Método: -")
        linea_fft.set_data([], [])
        canvas.draw()
    else:
        espectro = calcular_espectro(audio)
        etiqueta, metodo = buscar_coincidencia(espectro, base)

        if etiqueta:
            var_sonido.set(f"🔊 {etiqueta.upper()}")
            var_metodo.set(f"Método: {metodo}")
        else:
            espectro_mod = espectro[:num_features] if len(espectro) > num_features else np.pad(espectro, (0, num_features - len(espectro)))
            pred = modelo.predict([espectro_mod])[0]
            var_sonido.set(f"🔊 {pred.upper()}")
            var_metodo.set("Método: Clasificación ML")

        freqs = np.fft.rfftfreq(len(audio), 1 / FS)
        linea_fft.set_data(freqs, espectro[:len(freqs)])
        ax.set_xlim(0, 8000)
        ax.set_ylim(0, 1.05)
        canvas.draw()

    ventana.after(1500, detectar)

ventana.after(1000, detectar)
ventana.mainloop()
