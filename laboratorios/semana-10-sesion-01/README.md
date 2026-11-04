# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el pipeline de tu proyecto de la Semana 9, Sesión 2.

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

Archivo: [`starter/ejercicio1_linea_base_y_ensamblaje.py`](./starter/ejercicio1_linea_base_y_ensamblaje.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. El script ya trae la preparación de la Semana 9 y una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-01/starter
python3 ejercicio1_linea_base_y_ensamblaje.py
```

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

Solo o en pareja. Archivo: [`starter/ejercicio2_regularizacion.py`](./starter/ejercicio2_regularizacion.py), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_regularizacion.py
```

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

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el script tú.


### Entregable 1: `modelos_proyecto.py`

Un script, en la carpeta de tu proyecto, que entrene con tu pipeline:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

_(tu respuesta)_

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

_(tu respuesta)_

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

_(tu respuesta)_

**¿Cuánto mejora tu mejor modelo a la línea base?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

_(tu respuesta)_
