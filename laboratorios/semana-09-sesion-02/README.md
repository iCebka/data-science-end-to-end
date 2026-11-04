# Semana 9, Sesión 2: Pipeline reproducible de features

> Objetivo de la sesión: construir un pipeline de transformación reproducible con scikit-learn (imputación, codificación, escalado) para el dataset de tu proyecto, que evite la fuga de información entre entrenamiento y validación.

**Lectura previa de esta sesión:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

**Referencia de esta sesión:** [Pipelines and composite estimators (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/compose.html).

## Requisitos antes de esta sesión

scikit-learn ya viene en el Codespace. En tu laptop, instálalo con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import sklearn; print(sklearn.__version__)"
```

Este laboratorio usa `sparse_output=False` en `OneHotEncoder`, disponible desde scikit-learn 1.2. El `requirements.txt` del repositorio ya fija una versión posterior.

Ten a mano el script de features de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la fuga de información, y por qué un modelo con fuga puede verse excelente en la evaluación y fallar en producción?**

La fuga de información ocurre cuando el modelo utiliza datos que no debería conocer durante el entrenamiento, como información del conjunto de prueba o variables que no estarán disponibles al momento de predecir. Esto puede producir resultados demasiado buenos en la evaluación, porque el modelo aprovecha información que no tendría en condiciones reales. En producción, al no disponer de esa información, su rendimiento puede empeorar considerablemente.

**El artículo describe varias formas en que se produce una fuga. ¿Cuál de ellas viste la sesión pasada con la duración del viaje?**

Vimos un caso de target leakage, porque la duración real del viaje solo se conoce cuando este ha terminado. Si queremos predecir la tarifa antes de realizar el viaje, no podemos utilizar esa duración como una característica del modelo, aunque esté muy relacionada con el precio.

## Pregunta de apertura

**Para escalar la distancia, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?**

La media y la desviación estándar deben calcularse solo con los viajes de entrenamiento. Después, usamos esos mismos valores para escalar los datos de prueba. Si calculamos las estadísticas con todos los viajes, estaríamos incorporando información del conjunto de prueba al entrenamiento. Esto puede afectar la evaluación, aunque la diferencia en los resultados sea pequeña.

## Ejercicio 1: encontrar y corregir la fuga

Archivo: [`starter/ejercicio1_fuga.py`](./starter/ejercicio1_fuga.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv): los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. La primera parte del script es el código de un compañero: imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. La segunda parte es la versión correcta, con 3 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-02/starter
python3 ejercicio1_fuga.py
```

**Respuesta 1, antes de ver la solución:** en la versión del compañero, ¿en qué línea entra el conjunto de prueba al entrenamiento? ¿Qué número de la salida lo delata?

La fuga ocurre en la línea X_preparado = escalador.fit_transform(imputador.fit_transform(X)), porque el imputador y el escalador calculan sus estadísticas usando todo el dataset antes de dividirlo. El número que lo delata es 14875 en las filas que vio el escalador, cuando debería haber visto solo las 11900 filas de entrenamiento.

**Respuesta 2 (TODO 1 y 3):** sin resolver, la versión correcta dice que tiene 2975 filas de prueba. ¿Qué está midiendo ese error, si prueba y entrenamiento son las mismas filas?

Está midiendo el error sobre los mismos datos utilizados para entrenar el modelo. Por lo tanto, es un error de entrenamiento, aunque aparezca como error de prueba. No permite evaluar correctamente qué tan bien funcionaría el modelo con datos nuevos.

**Respuesta 3:** ¿cuánto esperas que cambie el error absoluto medio en prueba al corregir la fuga? Si casi no cambia, ¿significa que no había fuga?

Esperaría que el error cambie poco, porque las estadísticas de entrenamiento probablemente sean similares a las del dataset completo. Sin embargo, que el error casi no cambie no significa que no existiera fuga. El problema está en haber utilizado información del conjunto de prueba para preparar los datos, independientemente de cuánto afecte al resultado.

**Lo que confirmó la salida, TODO por TODO:**

```
s9w2e1  | Version del companero
s9w2e1  | Filas que vio el escalador: 2975.0
s9w2e1  | Media de distancia usada para escalar: 8.5034
s9w2e1  | Error absoluto medio en prueba: 2.5375
s9w2e1  | 
s9w2e1  | Version correcta
s9w2e1  | Filas que vio el escalador: 2380.0
s9w2e1  | Media de distancia usada para escalar: 0.0076
s9w2e1  | Filas de prueba: 595
s9w2e1  | Error absoluto medio en prueba: 2.5375
s9w2e1 exited with code 0

```

El escalador del compañero utilizó las 2975 filas del dataset, mientras que el de la versión corregida utilizó 2380 filas de entrenamiento y dejó 595 para prueba. Sin embargo, en el TODO 1 se dividió X_preparado en lugar de X, por lo que los datos ya estaban imputados y escalados antes de la división. Esto explica que la media de distancia del segundo escalador sea 0.0076 en lugar de 8.5034. Ambos modelos obtuvieron un MAE de 2.5375, pero la fuga todavía no está completamente corregida. Para solucionarla, el TODO 1 debe dividir X directamente, de modo que el pipeline aprenda las estadísticas únicamente con entrenamiento.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

ACTIVIDAD NO REALIZADA EN CLASE.

**Tu razonamiento:**

ACTIVIDAD NO REALIZADA EN CLASE.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

ACTIVIDAD NO REALIZADA EN CLASE.

## Ejercicio 2: el pipeline completo

Solo o en pareja. Archivo: [`starter/ejercicio2_pipeline.py`](./starter/ejercicio2_pipeline.py), sobre el mismo `viajes_features.csv`. Completa los 5 `TODO`: división, `ColumnTransformer` con tres grupos de columnas, `fit` solo con entrenamiento, verificación de las medianas, y un viaje desde una zona que el entrenamiento nunca vio. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_pipeline.py
```

```
s9w2e2  | Columnas despues de preparar: 25
s9w2e2  | Medianas aprendidas: [6.94 1.  ]
s9w2e2  | Medianas de entrenamiento: [6.94 1.  ]
s9w2e2  | Medianas de todo el dataset: [6.93 1.  ]
s9w2e2  | 
s9w2e2  | Comprobacion
s9w2e2  | TODO 1: OK
s9w2e2  | TODO 2: OK
s9w2e2  | TODO 3: OK
s9w2e2  | TODO 4: OK
s9w2e2  | TODO 5: OK
s9w2e2 exited with code 0
```

**Respuesta 1:** ¿cuántas columnas quedan después de preparar los datos? Explica de dónde sale ese número a partir de los tres grupos de columnas.

Quedan 25 columnas: 2 numéricas (distancia_km y pasajeros), 5 que ya estaban listas y 18 generadas al aplicar One-Hot Encoding a las 3 variables categóricas. Las numéricas se imputan y escalan, las categóricas se convierten en columnas binarias y las demás pasan sin cambios.

**Respuesta 2:** ¿las medianas que aprendió el imputador coinciden con las de entrenamiento o con las de todo el dataset? ¿Qué demuestra eso?

Coinciden con las de entrenamiento: 6.94 para distancia y 1 para pasajeros. En cambio, la mediana de distancia del dataset completo es 6.93. Esto demuestra que el imputador aprendió sus estadísticas únicamente con los datos de entrenamiento, evitando la fuga de información.

**Respuesta 3:** ¿qué le pasa al viaje desde el Bronx? ¿Qué habría pasado con `pd.get_dummies` aplicado por separado a ese viaje?

El pipeline procesa el viaje sin errores. Como Bronx no apareció en entrenamiento, todas las columnas correspondientes a la zona de origen quedan en 0. Si aplicáramos pd.get_dummies por separado, se crearía una columna para Bronx y faltarían las columnas de otras zonas, generando una estructura diferente a la que espera el modelo.

## Ejercicio 3: el pipeline de tu proyecto

Usa tu archivo con las 2 features de la sesión pasada y la plantilla [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py).

### Paso 1: clasifica tus columnas (5 minutos)

Escribe qué columnas son numéricas, cuáles categóricas y cuáles ya están listas, y cuál es el objetivo. Deja fuera toda columna que no exista al momento de predecir.

_(tu clasificación)_

### Paso 2: decide cómo dividir (5 minutos)

¿Tu modelo predice el futuro a partir de filas con fecha? Si sí, divide por fecha; si no, al azar. Justifica.

_(tu decisión)_

### Paso 3: construye y corre el pipeline (15 minutos)

Completa la plantilla y córrela. Anota la forma de los conjuntos de entrenamiento y prueba ya preparados, y la lista de columnas que produce.

_(lo que imprimió tu pipeline)_

## Antes de la próxima sesión

**Lectura:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

_(tu respuesta)_
