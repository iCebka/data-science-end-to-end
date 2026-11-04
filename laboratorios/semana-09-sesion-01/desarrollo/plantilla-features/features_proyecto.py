import pandas as pd

# TODO: la misma ruta que usaste en la exploracion de la Semana 8.
datos = pd.read_csv("TODO_ruta_del_archivo.csv")

# TODO: la columna que tu modelo va a predecir.
OBJETIVO = "TODO_columna_objetivo"

# TODO, feature 1: crea la columna a partir de una o mas columnas
# existentes.
# datos["TODO_feature_1"] = ...

# TODO, feature 2.
# datos["TODO_feature_2"] = ...

# Evidencia: compara el objetivo segun cada feature nueva.
# Si es numerica, su correlacion con el objetivo:
# print(datos[["TODO_feature_1", OBJETIVO]].corr())
# Si es categorica o binaria, el objetivo promedio por grupo:
# print(datos.groupby("TODO_feature_2")[OBJETIVO].mean())

print(datos.head())
