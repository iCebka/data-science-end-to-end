import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)

# TODO: reemplaza por la ruta del archivo que tu pipeline de la Semana 7
# deja listo para modelar (una tabla del warehouse exportada a CSV, o el
# CSV limpio de tu etapa de calidad).
datos = pd.read_csv("TODO_ruta_del_archivo.csv")

# TODO: reemplaza por el nombre de la columna que tu modelo va a predecir.
OBJETIVO = "TODO_columna_objetivo"

# Paso 1: resumen.
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2))

# Paso 2: distribucion de la variable objetivo.
plt.figure()
plt.hist(datos[OBJETIVO], bins=40)
plt.title(OBJETIVO)
plt.savefig("figuras/distribucion_objetivo.png")
plt.close()

# Paso 3: atipicos de la variable objetivo con la regla del IQR.
q1 = datos[OBJETIVO].quantile(0.25)
q3 = datos[OBJETIVO].quantile(0.75)
iqr = q3 - q1
fuera = datos[(datos[OBJETIVO] < q1 - 1.5 * iqr) | (datos[OBJETIVO] > q3 + 1.5 * iqr)]
print(f"Atipicos de {OBJETIVO} segun IQR: {len(fuera)}")
print(fuera.head(10))

# TODO, paso 4: elige la variable de entrada que esperas que mas influya
# en el objetivo y grafica su relacion con el (dispersion si es numerica,
# promedio del objetivo por grupo si es categorica).

# TODO, paso 5: agrega aqui las consultas que necesites para confirmar
# cada uno de tus 3 hallazgos (un filtro, un groupby, una correlacion).