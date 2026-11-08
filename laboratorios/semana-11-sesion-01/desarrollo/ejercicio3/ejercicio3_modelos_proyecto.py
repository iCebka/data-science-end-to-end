
"""Ejercicio 3: primeros modelos de depreciacion de futbolistas."""

import os
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Dataset generado en la Semana 9, Sesion 1.
datos = pd.read_csv("features_proyecto.csv")

OBJETIVO = "TARGET_DEPRECIACION_40"

NUMERICAS = [
    "EDAD_ANIOS",
    "HEIGHT_IN_CM",
    "MARKET_VALUE_IN_EUR",
    "VALOR_PREVIO_12M_EUR",
    "VARIACION_PREVIA_12M_PCT",
    "DESFASE_PREVIO_12M_DIAS",
    "PARTIDOS_180D",
    "MINUTES_PLAYED_180D",
    "GOALS_180D",
    "ASSISTS_180D",
    "PARTIDOS_365D",
    "MINUTES_PLAYED_365D",
    "GOALS_365D",
    "ASSISTS_365D",
    "CAIDA_PREVIA_20",
]

CATEGORICAS = [
    "FOOT",
    "POSITION",
    "SUB_POSITION",
    "COMPETITION_ID",
]

YA_LISTAS = [
    "ES_MAYOR_30",
    "TIENE_PARTIDOS_180D",
    "TIENE_PARTIDOS_365D",
]

# Permitir que el imputador procese los valores faltantes.
datos["CAIDA_PREVIA_20"] = datos["CAIDA_PREVIA_20"].astype(float)

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

# Misma division temporal que en la Semana 9, Sesion 2.
fechas = pd.to_datetime(datos["VALUATION_DATE"])

fecha_corte = fechas.quantile(0.8)
separacion = pd.Timedelta(days=430)

entrenamiento = fechas < fecha_corte - separacion
prueba = fechas >= fecha_corte

X_ent = X.loc[entrenamiento]
X_prueba = X.loc[prueba]

y_ent = y.loc[entrenamiento]
y_prueba = y.loc[prueba]

print("Fecha de corte:", fecha_corte.date())
print("Ultima fecha permitida para entrenamiento:",
      (fecha_corte - separacion).date())
print("Filas de entrenamiento:", len(X_ent))
print("Filas de prueba:", len(X_prueba))
print("Filas reservadas:",
      len(datos) - len(X_ent) - len(X_prueba))


# Preparacion de la Semana 9.
def preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(
                strategy="median",
                add_indicator=True
            )),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),

        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )),
        ]), CATEGORICAS),

        ("ya_listas", "passthrough", YA_LISTAS),
    ])


resultados = []


def evaluar(nombre, modelo):
    pipeline = Pipeline([
        ("preparar", preparacion()),
        ("modelo", modelo),
    ])

    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio

    pred_ent = pipeline.predict(X_ent)
    pred_prueba = pipeline.predict(X_prueba)

    bal_ent = balanced_accuracy_score(y_ent, pred_ent)
    bal_prueba = balanced_accuracy_score(y_prueba, pred_prueba)

    f1_ent = f1_score(y_ent, pred_ent, zero_division=0)
    f1_prueba = f1_score(y_prueba, pred_prueba, zero_division=0)

    precision_prueba = precision_score(
        y_prueba, pred_prueba, zero_division=0
    )
    recall_prueba = recall_score(
        y_prueba, pred_prueba, zero_division=0
    )

    resultados.append({
        "modelo": nombre,
        "balanced_acc_entrenamiento": round(bal_ent, 4),
        "balanced_acc_prueba": round(bal_prueba, 4),
        "brecha": round(bal_ent - bal_prueba, 4),
        "f1_entrenamiento": round(f1_ent, 4),
        "f1_prueba": round(f1_prueba, 4),
        "precision_prueba": round(precision_prueba, 4),
        "recall_prueba": round(recall_prueba, 4),
        "segundos": round(segundos, 3),
    })

    print(
        f"{nombre:24s} "
        f"entrenamiento {bal_ent:.4f}   "
        f"prueba {bal_prueba:.4f}   "
        f"F1 prueba {f1_prueba:.4f}   "
        f"precision {precision_prueba:.4f}   "
        f"recall {recall_prueba:.4f}"
    )

    return pipeline


# Modelo 1: linea base.
linea_base = DummyClassifier(strategy="most_frequent")
evaluar("linea base", linea_base)


# Modelo 2: regresion logistica con regularizacion L2.
logistica = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=500,
    random_state=42,
)   
evaluar("regresion logistica", logistica)


# Modelo 3: ensamblaje con boosting.
boosting = HistGradientBoostingClassifier(
    max_iter=100,
    max_leaf_nodes=15,
    learning_rate=0.08,
    l2_regularization=1.0,
    class_weight="balanced",
    random_state=42,
)
evaluar("gradient boosting", boosting)


# Guardar resultados.
tabla = pd.DataFrame(resultados)

os.makedirs("output/ejercicio3", exist_ok=True)
tabla.to_csv("output/ejercicio3/resultados_proyecto.csv", index=False)

print()
print(tabla.to_string(index=False))

mejor = tabla.loc[tabla["balanced_acc_prueba"].idxmax(), "modelo"]

print("\nMejor modelo segun balanced accuracy:", mejor)
