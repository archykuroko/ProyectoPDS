import numpy as np
import sounddevice as sd
import joblib
import json
import os
import time
from tkinter import Tk, Label, StringVar, ttk

# Configuración
FS = 44100
DURACION = 1.0  # segundos por bloque
UMBRAL_SILENCIO = 0.01
CARPETA_ESPECTROS = "base_sonidos"
MODELO_PATH = "modelo_sonidos_knn.pkl"
UMBRAL_COINCIDENCIA = 0.92

# --- Funciones ---
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
            ruta = os.path.join(CARPETA_ESPECTROS, archivo)
            with open(ruta, "r") as f:
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

# Cargar modelo y base
modelo = joblib.load(MODELO_PATH)
base = cargar_base()
num_features = modelo.n_features_in_

# --- Interfaz gráfica ---
ventana = Tk()
ventana.title("Detector de Sonidos - Proyecto Señales")
ventana.geometry("500x240")
ventana.resizable(False, False)
style = ttk.Style()
style.theme_use("clam")

var_sonido = StringVar(value="Esperando sonido...")
var_metodo = StringVar(value="Método: -")

Label(ventana, text="🎧 Detector de Sonidos", font=("Segoe UI", 20, "bold")).pack(pady=10)
Label(ventana, textvariable=var_sonido, font=("Segoe UI", 16), foreground="blue").pack(pady=10)
Label(ventana, textvariable=var_metodo, font=("Segoe UI", 12), foreground="gray").pack()

# --- Bucle de detección ---
def detectar():
    audio = sd.rec(int(DURACION * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    energia = np.sqrt(np.mean(audio ** 2))

    if energia < UMBRAL_SILENCIO:
        var_sonido.set("Silencio")
        var_metodo.set("Método: -")
    else:
        espectro = calcular_espectro(audio)
        etiqueta, metodo = buscar_coincidencia(espectro, base)

        if etiqueta:
            var_sonido.set(f"🔊 {etiqueta.upper()}")
            var_metodo.set(f"Método: {metodo}")
        else:
            espectro = espectro[:num_features] if len(espectro) > num_features else np.pad(espectro, (0, num_features - len(espectro)))
            pred = modelo.predict([espectro])[0]
            var_sonido.set(f"🔊 {pred.upper()}")
            var_metodo.set("Método: Clasificación ML")

    ventana.after(1500, detectar)

ventana.after(1000, detectar)
ventana.mainloop()
