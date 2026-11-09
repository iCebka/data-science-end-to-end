# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el script `pipeline_proyecto.py` de tu proyecto, de la Semana 9, Sesión 2.

Los laboratorios de hoy son notebooks: entrenar y comparar modelos es experimentar, y en un notebook vuelves a correr solo el modelo que cambiaste, sin reentrenar los demás. La preparación no se copia en el notebook: se importa del script, para que todos los modelos usen exactamente la misma.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

Un ensamblaje es una técnica que combina las predicciones de varios modelos para obtener un resultado más preciso y estable. En bagging, los modelos se entrenan de forma independiente sobre distintas muestras aleatorias de los datos y sus predicciones se combinan mediante promedios o votaciones. En boosting, los modelos se entrenan secuencialmente, de manera que cada nuevo modelo intenta corregir los errores acumulados de los anteriores. 
Medium

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

Porque los árboles pueden cometer errores diferentes y, al promediar sus predicciones, algunos de esos errores se compensan. Esto reduce la varianza y hace que el resultado sea menos sensible a las particularidades de los datos de entrenamiento. La mejora depende de que los árboles tengan suficiente diversidad y sus errores no estén demasiado correlacionados.

## Pregunta de apertura

**Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?**

El modelo probablemente está sufriendo sobreajuste (overfitting). Esto significa que se ha ajustado demasiado a los datos de entrenamiento, aprendiendo incluso patrones particulares o ruido que no se generalizan a datos nuevos. Por eso obtiene un error bajo en entrenamiento, pero su rendimiento empeora al evaluarlo con datos de prueba.

## Ejercicio 1: línea base, un árbol y un bosque

Notebook: [`starter/ejercicio1_linea_base_y_ensamblaje.ipynb`](./starter/ejercicio1_linea_base_y_ensamblaje.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. La preparación de la Semana 9 está en [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py) y el notebook la importa. Ya trae una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos, cada uno en su celda. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

Espero que la línea base tenga un error relativamente alto, porque siempre predice la mediana de las tarifas sin considerar las características de cada viaje. Sirve como punto de referencia para comprobar si los modelos más complejos realmente mejoran las predicciones. No podemos conocer el MAE exacto antes de ejecutarlo.

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

Espero que el árbol sin límite tenga un error de entrenamiento cercano a cero, porque puede ajustarse a casi todos los viajes individualmente. Sin embargo, en prueba espero un error considerablemente mayor, ya que puede memorizar particularidades de los datos de entrenamiento y sobreajustarse.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad del árbol a 8, espero que aumente un poco el error de entrenamiento, pero que disminuya el error de prueba y se reduzca la brecha entre ambos. Con un bosque de 200 árboles, espero obtener predicciones más estables y un menor error de prueba, porque promediar varios árboles ayuda a reducir la varianza y el sobreajuste, aunque no necesariamente elimina la brecha.

**Lo que confirmó la tabla, TODO por TODO:**

```python
# TODO 1: la linea base, un DummyRegressor que siempre predice la mediana
# de la tarifa de entrenamiento.
linea_base = DummyRegressor(strategy="median")
evaluar("linea base", linea_base)

# TODO 2: un arbol de decision (DecisionTreeRegressor) sin ningun limite,
# con random_state=42.
arbol = DecisionTreeRegressor(random_state=42)
evaluar("arbol sin limite", arbol)

# TODO 3: el mismo arbol, regularizado con max_depth=8.
arbol_limitado = DecisionTreeRegressor(max_depth=8, random_state=42)
evaluar("arbol max_depth=8", arbol_limitado)

# TODO 4: un bosque aleatorio (RandomForestRegressor) de 200 arboles, con
# random_state=42.
bosque = RandomForestRegressor(n_estimators=200, random_state=42)
evaluar("bosque aleatorio", bosque)
```

```
s10w1e1  | linea base                   entrenamiento 13.1998   prueba 12.5308
s10w1e1  | arbol sin limite             entrenamiento  0.0029   prueba  1.7143
s10w1e1  | arbol max_depth=8            entrenamiento  1.0098   prueba  1.2773
s10w1e1  | bosque aleatorio             entrenamiento  0.4744   prueba  1.2993
s10w1e1  | 
s10w1e1  |            modelo  mae_entrenamiento  mae_prueba  brecha  segundos
s10w1e1  |        linea base            13.1998     12.5308 -0.6690     0.017
s10w1e1  |  arbol sin limite             0.0029      1.7143  1.7114     0.013
s10w1e1  | arbol max_depth=8             1.0098      1.2773  0.2675     0.009
s10w1e1  |  bosque aleatorio             0.4744      1.2993  0.8250     0.903
s10w1e1 exited with code 0
```


La línea base obtuvo el mayor error, mientras que el árbol sin límite presentó un sobreajuste evidente: su MAE fue prácticamente cero en entrenamiento, pero aumentó a 1.7143 en prueba. Al limitar la profundidad a 8, la brecha disminuyó de 1.7114 a 0.2675 y el error de prueba también mejoró. El bosque aleatorio redujo el error de prueba respecto al árbol sin límite, pero no superó al árbol limitado.

El mejor modelo fue el árbol con max_depth=8, con un MAE de prueba de 1.2773, ligeramente inferior al 1.2993 del bosque aleatorio. Esto confirma que aumentar la complejidad del modelo no garantiza una mejor generalización.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

ACTIVIDAD NO REALIZADA EN CLASE.

**Tu razonamiento:**

ACTIVIDAD NO REALIZADA EN CLASE.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

ACTIVIDAD NO  REALIZADA EN CLASE.

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Notebook: [`starter/ejercicio2_regularizacion.ipynb`](./starter/ejercicio2_regularizacion.ipynb), sobre el mismo `viajes_features.csv` y con la misma preparación importada. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

```
s10w1e2  | Filas de entrenamiento: 2380
s10w1e2  | linea base                         entrenamiento 13.1998   prueba 12.5308
s10w1e2  | lineal grado 3                     entrenamiento  0.8892   prueba 22.7273
s10w1e2  | ridge grado 3                      entrenamiento  1.0272   prueba 1.3209
s10w1e2  | gradient boosting                  entrenamiento  1.0705   prueba 1.2750
s10w1e2  | 
s10w1e2  |            modelo  mae_entrenamiento  mae_prueba  brecha  segundos
s10w1e2  |        linea base            13.1998     12.5308 -0.6690     0.013
s10w1e2  |    lineal grado 3             0.8892     22.7273 21.8381     4.512
s10w1e2  |     ridge grado 3             1.0272      1.3209  0.2938     0.866
s10w1e2  | gradient boosting             1.0705      1.2750  0.2044     0.378
s10w1e2  | Columnas con interacciones de grado 3: 3275
s10w1e2  | Mejor modelo: gradient boosting
s10w1e2  | 
s10w1e2  | Comprobacion
s10w1e2  | Paso 1: OK
s10w1e2  | Paso 2: revisa
s10w1e2  | Paso 3: OK
s10w1e2  | Paso 4: OK
s10w1e2  | Paso 5: OK
s10w1e2 exited with code 0

```

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

La regresión recibe 3275 columnas después de generar las interacciones de grado 3, frente a 2380 filas de entrenamiento. Es decir, hay 895 características más que observaciones, lo que facilita el sobreajuste porque el modelo tiene demasiada flexibilidad respecto a los datos disponibles. Además, la matriz presenta dependencias lineales entre sus columnas. Esto ayuda a explicar por qué el MAE fue de 0.8892 en entrenamiento, pero aumentó a 22.7273 en prueba.

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

Con Ridge, el MAE de entrenamiento aumentó ligeramente de 0.8892 a 1.0272, mientras que el de prueba disminuyó de 22.7273 a 1.3209. Ridge aplica regularización L2, que penaliza los coeficientes grandes y reduce la sensibilidad del modelo a las particularidades de los datos. Por eso empeora un poco en entrenamiento, pero generaliza mucho mejor. La brecha disminuyó de 21.8381 a 0.2938.

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

El mejor modelo en prueba fue Gradient Boosting, con un MAE de 1.2750, frente al 12.5308 de la línea base. Esto representa una reducción de 11.2558 dólares en el error absoluto medio.

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

La brecha negativa de la línea base puede ocurrir porque los conjuntos de entrenamiento y prueba contienen viajes diferentes. Aunque el modelo siempre predice la mediana aprendida en entrenamiento, las tarifas de prueba pueden estar más cerca de ese valor. Por eso obtuvo un MAE de 12.5308 en prueba, menor que el 13.1998 de entrenamiento. No significa que el modelo haya aprendido a predecir mejor, sino que la muestra de prueba resultó más favorable para esa predicción constante.

**Sobre el Paso 2:** la brecha obtenida fue de 21.8381, mientras que la comprobación exige más de 100. Los resultados muestran sobreajuste, pero no satisfacen ese umbral específico. La implementación sigue siendo consistente con los TODO y no hace falta modificarla para justificar las respuestas.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el notebook tú.


### Entregable 1: `modelos_proyecto.py`

Un notebook, en la misma carpeta que tu `pipeline_proyecto.py` (`psets/proyecto/modelado/`), que importe de ahí la preparación y la división (`from pipeline_proyecto import cargar_y_dividir, construir_preparacion`) en vez de copiarlas, y que entrene:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento, escrita por el notebook. Antes de hacer commit, reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa, no de celdas corridas en momentos distintos.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

Utilizamos `DummyClassifier(strategy="most_frequent")` como línea base, porque nuestro problema es de clasificación binaria. Este modelo siempre predice la clase más frecuente del entrenamiento, que corresponde a los jugadores que no presentan una depreciación de al menos el 40%.

La línea base obtuvo una balanced accuracy de 0,5000 y un F1 de 0,0000, porque no identifica ninguna depreciación. Nos sirve como referencia para comprobar si los modelos que utilizan las características de los jugadores realmente mejoran la clasificación.

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

Los dos modelos entrenados presentan un rendimiento ligeramente superior en entrenamiento, pero las diferencias son pequeñas.

La regresión logística obtuvo una balanced accuracy de 0,6865 en entrenamiento y 0,6796 en prueba, con una brecha de 0,0069. Gradient Boosting obtuvo 0,7065 en entrenamiento y 0,6975 en prueba, con una brecha de 0,0089.

También observamos diferencias pequeñas en F1: de 0,3809 a 0,3777 para la regresión logística y de 0,4001 a 0,3959 para Gradient Boosting.

Por tanto, no encontramos evidencia de un sobreajuste importante en estos modelos. Ambos mantienen un rendimiento relativamente similar al evaluarse sobre observaciones posteriores a las utilizadas para entrenar.

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

En la regresión logística utilizamos regularización L2 con C=1.0, que penaliza los coeficientes grandes para reducir la complejidad del modelo.

En Gradient Boosting limitamos el número máximo de hojas de cada árbol a 15, utilizamos una tasa de aprendizaje de 0,08 y aplicamos regularización L2 con l2_regularization=1.0.

Con estas configuraciones obtuvimos brechas pequeñas entre entrenamiento y prueba: 0,0069 en la regresión logística y 0,0089 en Gradient Boosting. Esto es compatible con una generalización razonable, aunque no podemos atribuir directamente esos resultados a la regularización porque todavía no hemos comparado versiones sin estas restricciones.

**¿Cuánto mejora tu mejor modelo a la línea base?**

El mejor modelo fue Gradient Boosting, con una balanced accuracy de 0,6975 en prueba, frente al 0,5000 de la línea base. Esto representa una mejora de 0,1975, es decir, 19,75 puntos porcentuales.

Además, obtuvo un F1 de 0,3959 y un recall de 0,6747, mientras que la línea base obtuvo cero en ambas métricas.

Aunque el ensamblaje supera claramente a la línea base, su precision de 0,2801 muestra que todavía produce numerosos falsos positivos. Por tanto, el siguiente paso sería estudiar el equilibrio entre precision y recall antes de utilizarlo para predicciones reales.

```python
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
```

```
s10w1e3  | Fecha de corte: 2023-02-09
s10w1e3  | Ultima fecha permitida para entrenamiento: 2021-12-06
s10w1e3  | Filas de entrenamiento: 302189
s10w1e3  | Filas de prueba: 90018
s10w1e3  | Filas reservadas: 57871
s10w1e3  | linea base               entrenamiento 0.5000   prueba 0.5000   F1 prueba 0.0000   precision 0.0000   recall 0.0000
s10w1e3  | regresion logistica      entrenamiento 0.6865   prueba 0.6796   F1 prueba 0.3777   precision 0.2673   recall 0.6438
s10w1e3  | gradient boosting        entrenamiento 0.7065   prueba 0.6975   F1 prueba 0.3959   precision 0.2801   recall 0.6747
s10w1e3  | 
s10w1e3  |              modelo  balanced_acc_entrenamiento  balanced_acc_prueba  brecha  f1_entrenamiento  f1_prueba  precision_prueba  recall_prueba  segundos
s10w1e3  |          linea base                      0.5000               0.5000  0.0000            0.0000     0.0000            0.0000         0.0000     0.860
s10w1e3  | regresion logistica                      0.6865               0.6796  0.0069            0.3809     0.3777            0.2673         0.6438     4.722
s10w1e3  |   gradient boosting                      0.7065               0.6975  0.0089            0.4001     0.3959            0.2801         0.6747    14.269
s10w1e3  | 
s10w1e3  | Mejor modelo segun balanced accuracy: gradient boosting
s10w1e3 exited with code 0
```

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error, mediría el tiempo de entrenamiento y de respuesta, el costo computacional y económico, y los recursos que necesita cada alternativa. También consideraría qué tan fácil es explicar sus predicciones, mantener el modelo y actualizarlo con nuevos datos.

Para una app de taxis es importante obtener estimaciones rápidas, consistentes y económicas. Por eso, si un ensamblaje ofrece un error similar al de una red neuronal o un LLM, pero necesita menos recursos y responde más rápido, probablemente sería la opción más conveniente.