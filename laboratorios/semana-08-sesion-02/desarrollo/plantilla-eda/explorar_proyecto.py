
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.makedirs("output/figuras", exist_ok=True)

datos = pd.read_csv("output/modelo_etiquetado.csv")

OBJETIVO = "TARGET_DEPRECIACION_40"
VALOR = "MARKET_VALUE_IN_EUR"
MINUTOS = "MINUTES_PLAYED_365D"

# Paso 1: resumen.

print("Paso 1: resumen")
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2))
print("Duplicados:", datos.duplicated(["PLAYER_ID", "VALUATION_DATE"]).sum())


# Paso 2: distribucion de la variable objetivo.

print("\nPaso 2: distribucion del objetivo")

print(datos[OBJETIVO].value_counts())
print(datos[OBJETIVO].value_counts(normalize=True).mul(100).round(2))

plt.figure()
datos[OBJETIVO].value_counts().sort_index().plot.bar()
plt.title("Distribucion de la depreciacion")
plt.xlabel("Depreciacion mayor o igual al 40% (0 = No, 1 = Si)")
plt.ylabel("Numero de observaciones")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output/figuras/distribucion_objetivo.png")
plt.close()


# Paso 3: distribucion y atipicos del valor de mercado.

print("\nPaso 3: atipicos del valor de mercado")

q1 = datos[VALOR].quantile(0.25)
q3 = datos[VALOR].quantile(0.75)
iqr = q3 - q1

limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

atipicos = datos[
    (datos[VALOR] < limite_inferior) |
    (datos[VALOR] > limite_superior)
]

print("Limite inferior:", limite_inferior)
print("Limite superior:", limite_superior)
print("Atipicos:", len(atipicos))
print("Porcentaje:", round(len(atipicos) / len(datos) * 100, 2))
print(atipicos[[VALOR, OBJETIVO]].head(10))

plt.figure()
plt.hist(np.log1p(datos[VALOR]), bins=50)
plt.title("Distribucion del valor de mercado")
plt.xlabel("log(1 + valor en EUR)")
plt.ylabel("Numero de observaciones")
plt.savefig("output/figuras/hist_valor.png")
plt.close()

plt.figure()
plt.boxplot(np.log1p(datos[VALOR].dropna()))
plt.title("Valor de mercado, diagrama de caja")
plt.ylabel("log(1 + valor en EUR)")
plt.savefig("output/figuras/boxplot_valor.png")
plt.close()


# Paso 4: relacion entre minutos jugados y depreciacion.

print("\nPaso 4: minutos jugados y depreciacion")

# Solo observaciones con minutos registrados.
con_minutos = datos.dropna(subset=[MINUTOS]).copy()

print("Observaciones con minutos:", len(con_minutos))

plt.figure()
plt.hist(con_minutos[MINUTOS], bins=50)
plt.title("Distribucion de minutos jugados en 365 dias")
plt.xlabel("Minutos jugados")
plt.ylabel("Numero de observaciones")
plt.savefig("output/figuras/hist_minutos.png")
plt.close()

plt.figure()
plt.boxplot([
    con_minutos[con_minutos[OBJETIVO] == 0][MINUTOS],
    con_minutos[con_minutos[OBJETIVO] == 1][MINUTOS]
])
plt.xticks([1, 2], ["Sin depreciacion", "Con depreciacion"])
plt.title("Minutos jugados segun depreciacion futura")
plt.ylabel("Minutos en los ultimos 365 dias")
plt.savefig("output/figuras/boxplot_minutos.png")
plt.close()

# Agrupar minutos en intervalos de 450.
bordes = [0, 450, 900, 1350, 1800, 2250, 2700, 3150, 3600, np.inf]

con_minutos["RANGO_MINUTOS"] = pd.cut(
    con_minutos[MINUTOS],
    bins=bordes,
    right=False
)

relacion = con_minutos.groupby(
    "RANGO_MINUTOS", observed=True
)[OBJETIVO].agg(["count", "mean"])

print("\nDepreciacion por rango de minutos:")
print(relacion.round(3))

plt.figure()
plt.bar(
    relacion.index.astype(str),
    relacion["mean"] * 100
)
plt.title("Depreciacion futura segun minutos jugados")
plt.xlabel("Minutos jugados en 365 dias")
plt.ylabel("Porcentaje de depreciaciones (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/figuras/minutos_vs_objetivo.png")
plt.close()


# Paso 5: consultas para confirmar tres hallazgos.

print("\nPaso 5: hallazgos")

# Hallazgo 1: distribucion de clases.
print("\nHallazgo 1: desbalance del objetivo")
print(datos[OBJETIVO].value_counts(normalize=True).mul(100).round(2))

# Hallazgo 2: cobertura de partidos.
print("\nHallazgo 2: cobertura deportiva")
print(datos["TIENE_PARTIDOS_365D"].value_counts())
print(datos.groupby("TIENE_PARTIDOS_365D")[OBJETIVO].mean().mul(100).round(2))

# Hallazgo 3: relacion de minutos y depreciacion.
print("\nHallazgo 3: minutos y depreciacion")
print(con_minutos.groupby(OBJETIVO)[MINUTOS].agg(
    ["count", "mean", "median"]
).round(2))

print("Correlacion minutos-target:",
      con_minutos[MINUTOS].corr(con_minutos[OBJETIVO]))

print("\nFiguras guardadas en output/figuras/")
