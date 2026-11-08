import os

import pandas as pd

# La misma tabla preparada en la Semana 8.
datos = pd.read_csv("modelo_etiquetado.csv")

# La columna que nuestro modelo va a predecir.
OBJETIVO = "TARGET_DEPRECIACION_40"

# Feature 1: jugador con al menos 30 años.
datos["ES_MAYOR_30"] = (datos["EDAD_ANIOS"] >= 30).astype(int)

# Feature 2: depreciacion previa de al menos el 20%.
datos["CAIDA_PREVIA_20"] = (
    datos["VARIACION_PREVIA_12M_PCT"] <= -20
).astype("Int64")

# Mantener los valores faltantes cuando no existe una valoracion previa.
datos.loc[
    datos["VARIACION_PREVIA_12M_PCT"].isna(),
    "CAIDA_PREVIA_20"
] = pd.NA

# Evidencia: comparar la depreciacion futura por grupos.
print("Depreciacion segun edad:")
print(
    datos.groupby("ES_MAYOR_30")[OBJETIVO]
    .agg(["count", "mean"])
    .round(3)
)

print("\nDepreciacion segun caida previa:")
print(
    datos.groupby("CAIDA_PREVIA_20", dropna=False)[OBJETIVO]
    .agg(["count", "mean"])
    .round(3)
)

print("\nPrimeras filas:")
print(datos.head())



# Guardar el dataset con las dos columnas nuevas.
os.makedirs("output/ejercicio3", exist_ok=True)
datos.to_csv("output/ejercicio3/features_proyecto.csv", index=False)