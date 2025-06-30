import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from collections import Counter

# Ruta de tu base
CARPETA_BASE = "base_sonidos"

# Cargar espectros
X = []
y = []

for archivo in os.listdir(CARPETA_BASE):
    if archivo.endswith(".json"):
        ruta = os.path.join(CARPETA_BASE, archivo)
        with open(ruta, "r") as f:
            data = json.load(f)
            magnitudes = np.array(data["magnitudes"])
            etiqueta = data["etiqueta"]
            X.append(magnitudes)
            y.append(etiqueta)

# Recortar todos los vectores al más corto
min_len = min(len(vec) for vec in X)
X = [vec[:min_len] for vec in X]

# Convertir a array
X = np.array(X)
y = np.array(y)

# 🧹 Filtrar clases con menos de 2 muestras
conteo = Counter(y)
X_filtrado = []
y_filtrado = []

for i in range(len(y)):
    if conteo[y[i]] > 1:
        X_filtrado.append(X[i])
        y_filtrado.append(y[i])

X = np.array(X_filtrado)
y = np.array(y_filtrado)

# Separar en entrenamiento y prueba (ya se puede estratificar)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Entrenar modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar
y_pred = modelo.predict(X_test)
print("📊 Reporte de clasificación (Random Forest):")
print(classification_report(y_test, y_pred))

# Guardar modelo
joblib.dump(modelo, "modelo_sonidos_rf.pkl")
print("✅ Modelo guardado como 'modelo_sonidos_rf.pkl'")
