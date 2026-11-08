
"""Ejercicio 3: ensamblaje, red neuronal y LLM para depreciacion."""

import os
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


datos = pd.read_csv("features_proyecto.csv")
OBJETIVO = "TARGET_DEPRECIACION_40"
SALIDA = "output/ejercicio3"
os.makedirs(SALIDA, exist_ok=True)

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

# Variables que mostraremos al LLM.
LEGIBLES = [
    "EDAD_ANIOS",
    "POSITION",
    "SUB_POSITION",
    "MARKET_VALUE_IN_EUR",
    "VALOR_PREVIO_12M_EUR",
    "VARIACION_PREVIA_12M_PCT",
    "ES_MAYOR_30",
    "CAIDA_PREVIA_20",
    "PARTIDOS_365D",
    "MINUTES_PLAYED_365D",
    "GOALS_365D",
    "ASSISTS_365D",
]

datos["CAIDA_PREVIA_20"] = datos["CAIDA_PREVIA_20"].astype(float)

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

# Misma division que en las semanas 9 y 10.
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
print("Entrenamiento:", len(X_ent))
print("Prueba:", len(X_prueba))


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


# Muestra de 10: 5 positivos y 5 negativos.
# Es una muestra diagnostica, no representativa.
indices_muestra = pd.concat([
    y_prueba[y_prueba == 0].sample(5, random_state=7),
    y_prueba[y_prueba == 1].sample(5, random_state=7),
]).sample(frac=1, random_state=7).index

muestra = X_prueba.loc[indices_muestra]
y_muestra = y_prueba.loc[indices_muestra]


def metricas(y_real, pred):
    return {
        "balanced_accuracy": balanced_accuracy_score(y_real, pred),
        "f1": f1_score(y_real, pred, zero_division=0),
        "precision": precision_score(y_real, pred, zero_division=0),
        "recall": recall_score(y_real, pred, zero_division=0),
    }


resultados = []
predicciones_muestra = pd.DataFrame({
    "target_real": y_muestra.to_numpy()
})


def evaluar(nombre, modelo, balancear=False):
    pipeline = Pipeline([
        ("preparar", preparacion()),
        ("modelo", modelo),
    ])

    inicio = time.perf_counter()

    if balancear:
        pesos = compute_sample_weight(
            class_weight="balanced", y=y_ent
        )
        pipeline.fit(
            X_ent, y_ent, modelo__sample_weight=pesos
        )
    else:
        pipeline.fit(X_ent, y_ent)

    segundos = time.perf_counter() - inicio

    pred_prueba = pipeline.predict(X_prueba)
    pred_muestra = pipeline.predict(muestra)

    completo = metricas(y_prueba, pred_prueba)
    reducido = metricas(y_muestra, pred_muestra)

    resultados.append({
        "modelo": nombre,
        "balanced_acc_prueba": round(
            completo["balanced_accuracy"], 4
        ),
        "balanced_acc_10": round(
            reducido["balanced_accuracy"], 4
        ),
        "f1_10": round(reducido["f1"], 4),
        "precision_10": round(reducido["precision"], 4),
        "recall_10": round(reducido["recall"], 4),
        "segundos_entrenamiento": round(segundos, 3),
    })

    predicciones_muestra[nombre] = pred_muestra

    print(
        f"{nombre:20s} "
        f"balanced acc prueba {completo['balanced_accuracy']:.4f} "
        f"balanced acc 10 {reducido['balanced_accuracy']:.4f} "
        f"F1 10 {reducido['f1']:.4f} "
        f"tiempo {segundos:.2f} s"
    )

    return pipeline


# Modelo 1: mejor ensamblaje de la sesion pasada.
ensamblaje = evaluar(
    "gradient boosting",
    HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.08,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    ),
)

# Modelo 2: red neuronal simple con una capa de 64 neuronas.
red = evaluar(
    "red neuronal",
    MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=200,
        batch_size=512,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
    ),
    balancear=True,
)

modelo_red = red.named_steps["modelo"]

parametros_red = (
    sum(matriz.size for matriz in modelo_red.coefs_)
    + sum(vector.size for vector in modelo_red.intercepts_)
)

print("\nParametros aprendidos por la red:", parametros_red)
print("Iteraciones de la red:", modelo_red.n_iter_)


# Generar 30 ejemplos historicos para el LLM:
# 15 casos positivos y 15 negativos.
indices_ejemplos = pd.concat([
    y_ent[y_ent == 0].sample(15, random_state=7),
    y_ent[y_ent == 1].sample(15, random_state=7),
]).sample(frac=1, random_state=7).index

ejemplos = X_ent.loc[indices_ejemplos, LEGIBLES].copy()
ejemplos[OBJETIVO] = y_ent.loc[indices_ejemplos]

sin_etiqueta = muestra[LEGIBLES].copy()

prompt = (
    "Eres un clasificador de depreciacion del valor de mercado "
    "de futbolistas.\n"
    "Predice si el valor de mercado disminuira al menos un 40% "
    "aproximadamente 12 meses despues de la valoracion actual "
    "(con tolerancia de 65 dias).\n\n"
    "La etiqueta es 1 si ocurre esa depreciacion y 0 si no ocurre. "
    "Las variables describen informacion conocida en la fecha "
    "de valoracion. Los valores vacios indican datos faltantes.\n\n"
    "Los ejemplos historicos estan balanceados deliberadamente: "
    "no representan la frecuencia real de depreciaciones.\n"
    "Responde solamente con 10 numeros, cada uno 0 o 1, "
    "uno por linea y en el mismo orden. No agregues texto.\n\n"
    "Ejemplos historicos con etiqueta:\n"
    + ejemplos.to_csv(index=False, na_rep="")
    + "\nFutbolistas sin etiqueta:\n"
    + sin_etiqueta.to_csv(index=False, na_rep="")
)

ruta_prompt = f"{SALIDA}/prompt_llm.txt"
with open(ruta_prompt, "w", encoding="utf-8") as archivo:
    archivo.write(prompt)

print("\nPrompt generado:", ruta_prompt)
print("Caracteres:", len(prompt))


# Leer las 10 predicciones si ya existe el archivo.
ruta_respuestas = f"{SALIDA}/respuestas_llm.txt"

if os.path.exists(ruta_respuestas):
    with open(ruta_respuestas, encoding="utf-8") as archivo:
        pred_llm = [
            int(linea.strip())
            for linea in archivo
            if linea.strip()
        ]

    if len(pred_llm) != 10 or any(
        pred not in (0, 1) for pred in pred_llm
    ):
        raise ValueError(
            "respuestas_llm.txt debe contener exactamente "
            "10 numeros, cada uno 0 o 1."
        )

    medidas_llm = metricas(y_muestra, pred_llm)

    resultados.append({
        "modelo": "LLM",
        "balanced_acc_prueba": None,
        "balanced_acc_10": round(
            medidas_llm["balanced_accuracy"], 4
        ),
        "f1_10": round(medidas_llm["f1"], 4),
        "precision_10": round(medidas_llm["precision"], 4),
        "recall_10": round(medidas_llm["recall"], 4),
        "segundos_entrenamiento": None,
    })

    predicciones_muestra["LLM"] = pred_llm
    print(
        "\nLLM balanced acc 10:",
        round(medidas_llm["balanced_accuracy"], 4)
    )
else:
    print(
        "\nLLM pendiente. Copia el prompt al chat y guarda "
        "sus 10 respuestas en:", ruta_respuestas
    )


# Guardar resultados y predicciones para auditoria.
tabla = pd.DataFrame(resultados)
tabla.to_csv(f"{SALIDA}/resultados_comparacion.csv", index=False)

predicciones_muestra.to_csv(
    f"{SALIDA}/predicciones_10.csv", index=False
)

print("\nComparacion:")
print(tabla.to_string(index=False))
