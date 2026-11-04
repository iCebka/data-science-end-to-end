import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TODO: el archivo de tu proyecto, con las 2 features de la sesion pasada
# ya agregadas (o el script que las agrega, antes de esta linea).
datos = pd.read_csv("TODO_ruta_del_archivo.csv")

OBJETIVO = "TODO_columna_objetivo"
NUMERICAS = ["TODO_numerica_1", "TODO_numerica_2"]
CATEGORICAS = ["TODO_categorica_1"]
YA_LISTAS = []

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

# TODO: decide como dividir. Si tus filas tienen fecha y vas a predecir
# el futuro, divide por fecha (entrenamiento antes de una fecha de corte,
# prueba despues) en vez de al azar.
X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

preparacion = ColumnTransformer([
    ("numericas", Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ]), NUMERICAS),
    ("categoricas", Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAS),
    ("ya_listas", "passthrough", YA_LISTAS),
])

X_ent_listo = preparacion.fit_transform(X_ent)
X_prueba_listo = preparacion.transform(X_prueba)
print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
print(list(preparacion.get_feature_names_out()))
