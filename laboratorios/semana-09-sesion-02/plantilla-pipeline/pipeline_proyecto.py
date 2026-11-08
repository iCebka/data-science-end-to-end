
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Dataset de la Semana 9, Sesion 1.
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

# CAIDA_PREVIA_20 contiene valores faltantes.
# Convertirla a float para que el imputador pueda procesarla.
datos["CAIDA_PREVIA_20"] = datos["CAIDA_PREVIA_20"].astype(float)

# Paso 1: separar las entradas y el objetivo.
X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

# Paso 2: division temporal.
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
print("Filas reservadas entre ambos periodos:",
      len(datos) - len(X_ent) - len(X_prueba))

# Paso 3: pipeline de preparacion.
preparacion = ColumnTransformer([
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

# Paso 4: aprender transformaciones solo de entrenamiento.
X_ent_listo = preparacion.fit_transform(X_ent)

# Aplicar las mismas transformaciones a prueba.
X_prueba_listo = preparacion.transform(X_prueba)

print("\nEntrenamiento:", X_ent_listo.shape)
print("Prueba:", X_prueba_listo.shape)

print("\nColumnas despues de preparar:")
print(list(preparacion.get_feature_names_out()))

print("\nTargets:")
print("Entrenamiento:")
print(y_ent.value_counts())
print("Prueba:")
print(y_prueba.value_counts())
