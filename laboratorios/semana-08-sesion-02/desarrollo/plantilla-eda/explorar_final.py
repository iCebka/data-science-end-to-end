"""Semana 8, sesion 2, ejercicio 3: EDA del proyecto Transfermarkt.

Ejecutar desde desarrollo/ o plantilla-eda/:
    python plantilla-eda/explorar_proyecto.py
    python explorar_proyecto.py

Genera figuras en output/figuras/ dentro de plantilla-eda/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
FIGURAS = CARPETA / "output" / "figures"
FIGURAS.mkdir(parents=True, exist_ok=True)

# Aceptar tanto la estructura actual como la salida del pipeline organizado.
RUTAS = [
    CARPETA / "output" / "modelo_etiquetado.csv"
]
CSV = next((ruta for ruta in RUTAS if ruta.exists()), None)
if CSV is None:
    raise FileNotFoundError(f"No se encontro modelo_etiquetado.csv en: {RUTAS}")

datos = pd.read_csv(CSV)
OBJETIVO = "TARGET_DEPRECIACION_40"
VALOR = "MARKET_VALUE_IN_EUR"
MINUTOS = "MINUTES_PLAYED_365D"
PREVIA = "VARIACION_PREVIA_12M_PCT"

# Paso 1: resumen.
print("Paso 1: resumen")
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print("Valores faltantes:")
print(datos.isna().sum())
print(datos.describe().round(2))
print("Duplicados jugador-fecha:", datos.duplicated(["PLAYER_ID", "VALUATION_DATE"]).sum())

# Paso 2: distribucion de la variable objetivo.
print("\nPaso 2: distribucion del objetivo")
print(datos[OBJETIVO].value_counts())
print(datos[OBJETIVO].value_counts(normalize=True).mul(100).round(2))

plt.figure()
datos[OBJETIVO].value_counts().sort_index().plot.bar()
plt.title("Distribucion de la depreciacion futura")
plt.xlabel("Depreciacion mayor o igual al 40% (0 = No, 1 = Si)")
plt.ylabel("Numero de observaciones")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURAS / "distribucion_objetivo.png")
plt.close()

# Paso 3: atipicos. El target es binario: usamos el valor de mercado.
print("\nPaso 3: atipicos del valor de mercado")
q1 = datos[VALOR].quantile(0.25)
q3 = datos[VALOR].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr
atipicos = datos[(datos[VALOR] < limite_inferior) | (datos[VALOR] > limite_superior)]
print("Limite inferior:", limite_inferior)
print("Limite superior:", limite_superior)
print("Atipicos:", len(atipicos))
print("Porcentaje:", round(len(atipicos) / len(datos) * 100, 2))
print(atipicos[[VALOR, OBJETIVO]].head(10))

plt.figure()
plt.hist(np.log1p(datos[VALOR].dropna()), bins=50)
plt.title("Distribucion del valor de mercado")
plt.xlabel("log(1 + valor en EUR)")
plt.ylabel("Numero de observaciones")
plt.savefig(FIGURAS / "hist_valor.png")
plt.close()

plt.figure()
plt.boxplot(np.log1p(datos[VALOR].dropna()))
plt.title("Valor de mercado, diagrama de caja")
plt.ylabel("log(1 + valor en EUR)")
plt.savefig(FIGURAS / "boxplot_valor.png")
plt.close()

# Paso 4: edad como variable principal asociada al objetivo.
print("\nPaso 4: edad y depreciacion")
datos["GRUPO_EDAD"] = pd.cut(
    datos["EDAD_ANIOS"],
    bins=[14, 20, 25, 30, 35, 40, 60],
    labels=["14-20", "20-25", "25-30", "30-35", "35-40", "40-60"],
    include_lowest=True,
)
por_edad = datos.groupby("GRUPO_EDAD", observed=True)[OBJETIVO].agg(["count", "mean"])
print(por_edad.assign(porcentaje=por_edad["mean"] * 100).round(2))

plt.figure()
plt.bar(por_edad.index.astype(str), por_edad["mean"] * 100)
plt.title("Depreciacion futura segun edad")
plt.xlabel("Edad del jugador")
plt.ylabel("Depreciaciones (%)")
plt.savefig(FIGURAS / "edad_vs_objetivo.png")
plt.close()

# Paso 5: tres hallazgos, con consultas y graficos sencillos.
print("\nPaso 5: hallazgos")

# Hallazgo 1: los minutos tienen relacion debil y no lineal con el target.
print("\nHallazgo 1: minutos jugados y depreciacion")
con_minutos = datos.dropna(subset=[MINUTOS]).copy()
print("Observaciones con minutos:", len(con_minutos))
print(con_minutos.groupby(OBJETIVO)[MINUTOS].agg(["count", "mean", "median"]).round(2))
print("Correlacion minutos-target:", con_minutos[MINUTOS].corr(con_minutos[OBJETIVO]))
con_minutos["RANGO_MINUTOS"] = pd.cut(
    con_minutos[MINUTOS],
    bins=[0, 450, 900, 1350, 1800, 2250, 2700, 3150, 3600, np.inf],
    right=False,
)
por_minutos = con_minutos.groupby("RANGO_MINUTOS", observed=True)[OBJETIVO].agg(["count", "mean"])
print(por_minutos.assign(porcentaje=por_minutos["mean"] * 100).round(2))

plt.figure()
plt.bar(por_minutos.index.astype(str), por_minutos["mean"] * 100)
plt.title("Depreciacion futura segun minutos jugados")
plt.xlabel("Minutos jugados en los ultimos 365 dias")
plt.ylabel("Depreciaciones (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURAS / "minutos_vs_objetivo.png")
plt.close()

# Hallazgo 2: asociacion entre variacion previa y depreciacion futura.
print("\nHallazgo 2: variacion previa del valor")
print("Sin variacion previa:", datos[PREVIA].isna().sum())
print("Correlacion variacion previa-target:", datos[PREVIA].corr(datos[OBJETIVO]))
datos["GRUPO_VARIACION_PREVIA"] = pd.cut(
    datos[PREVIA],
    bins=[-np.inf, -50, -20, 0, 20, np.inf],
    labels=["Caida >50%", "Caida 20-50%", "Caida hasta 20%", "Aumento hasta 20%", "Aumento >20%"],
)
por_variacion = datos.groupby("GRUPO_VARIACION_PREVIA", observed=True)[OBJETIVO].agg(["count", "mean"])
print(por_variacion.assign(porcentaje=por_variacion["mean"] * 100).round(2))

plt.figure()
plt.bar(por_variacion.index.astype(str), por_variacion["mean"] * 100)
plt.title("Depreciacion futura segun variacion historica")
plt.ylabel("Depreciaciones (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIGURAS / "variacion_previa_vs_objetivo.png")
plt.close()

# Hallazgo 3: cobertura de los registros deportivos.
print("\nHallazgo 3: cobertura deportiva")
print(datos["TIENE_PARTIDOS_365D"].value_counts())
print("Sin minutos:", datos[MINUTOS].isna().sum())
print("Porcentaje sin minutos:", round(datos[MINUTOS].isna().mean() * 100, 2))
print("Depreciacion por cobertura:")
print(datos.groupby("TIENE_PARTIDOS_365D")[OBJETIVO].agg(["count", "mean"]).round(3))
print("No tener registros de partidos no equivale a haber jugado cero minutos.")

print("\nFiguras guardadas en:", FIGURAS)
