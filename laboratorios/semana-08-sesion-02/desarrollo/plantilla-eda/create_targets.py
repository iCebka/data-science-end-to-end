import pandas as pd

PATH = 'output'
CSV_PATH = PATH + "/valuaciones_h12m_tol65d.csv"

# Parámetro configurable: porcentaje mínimo de depreciación.
UMBRAL_DEPRECIACION = 40

df = pd.read_csv(CSV_PATH)

# Construir el nombre del target dinámicamente.
OBJETIVO = f"TARGET_DEPRECIACION_{UMBRAL_DEPRECIACION}"

# Calcular el límite de proporción correspondiente.
limite = 1 - UMBRAL_DEPRECIACION / 100

# Target binario.
df[OBJETIVO] = (
    df["PROPORCION_VALOR"] <= limite
).astype("Int64")

# Conservar los valores desconocidos.
df.loc[df["PROPORCION_VALOR"].isna(), OBJETIVO] = pd.NA

# Exportar tabla maestra.
df.to_csv(f"{PATH}/valuaciones_con_target.csv", index=False)

# Separar observaciones etiquetadas y sin etiqueta.
etiquetadas = df.dropna(subset=[OBJETIVO]).copy()
sin_etiqueta = df[df[OBJETIVO].isna()].copy()

etiquetadas.to_csv(f"{PATH}/valuaciones_etiquetadas_u{UMBRAL_DEPRECIACION}.csv", index=False)
sin_etiqueta.to_csv(f"{PATH}/valuaciones_sin_etiqueta_u{UMBRAL_DEPRECIACION}.csv", index=False)

# Resumen.
print(f"Umbral de depreciación: {UMBRAL_DEPRECIACION}%")
print(f"Proporción límite: {limite:.2f}")
print(f"Columna objetivo: {OBJETIVO}")
print(f"Etiquetadas: {len(etiquetadas):,}")
print(f"Sin etiqueta: {len(sin_etiqueta):,}")

print("\nDistribución del target:")
print(etiquetadas[OBJETIVO].value_counts())

print("\nDistribución porcentual:")
print(
    etiquetadas[OBJETIVO]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)