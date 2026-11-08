# Semana 11, Sesión 1: Gestión y trazabilidad de experimentos: eligiendo el modelo ganador con evidencia

> Objetivo de la sesión: registrar de forma trazable los parámetros, métricas y artefactos de las corridas de entrenamiento de las dos sesiones anteriores, y elegir el modelo final de tu proyecto con esa evidencia.

**Lectura previa de esta sesión:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

**Referencia de esta sesión:** [ML Experiment Tracking (documentación oficial de MLflow)](https://mlflow.org/docs/latest/ml/tracking/).

## Requisitos antes de esta sesión

Hoy se agrega una herramienta: MLflow, que registra cada corrida de entrenamiento. Como en un equipo de datos, el registro vive en un servidor de seguimiento (tracking server), no en tu notebook: el Codespace levanta uno en el puerto 5050 cada vez que inicia, y los notebooks lo encuentran por la variable de entorno `MLFLOW_TRACKING_URI`.

Si tu Codespace se creó antes de esta sesión, sincroniza tu fork y reconstruye el contenedor (`Ctrl+Shift+P` o `Cmd+Shift+P` → "Codespaces: Rebuild Container"; en tu laptop, "Dev Containers: Rebuild Container"). Reconstruir instala MLflow y configura el servidor; tus archivos no se pierden. Al terminar, comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Las últimas líneas deben decir `mlflow, version 2.17.2` y `servidor de MLflow: responde en http://127.0.0.1:5050`. Si el servidor no responde (por ejemplo, después de detener y volver a abrir el Codespace sin que arranque), levántalo a mano con `bash .devcontainer/mlflow-server.sh`.

La interfaz de MLflow se abre desde la pestaña **Ports** de VS Code: el puerto 5050, "MLflow (Semana 11)", con el ícono del globo. Ahí ves cada experimento, sus corridas, y puedes compararlas lado a lado.

Los laboratorios de hoy son notebooks, como los de la Semana 10, y la preparación se importa de `preparacion_taxis.py`. El registro (la base de datos del servidor y los modelos guardados) queda en la carpeta `.mlflow/` de la raíz del repositorio, que no se sube (está en `.gitignore`): lo que entregas es el notebook y la tabla de corridas que exporta.

**Si trabajas en tu laptop sin el contenedor (Opción B del README):** instala con `pip install -r requirements.txt` y, en una terminal aparte desde la raíz de tu fork, levanta el mismo servidor con `mlflow server --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination .mlflow/artefactos --port 5050`. Antes de abrir los notebooks, define la variable `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`. Sin esa variable, MLflow guarda el registro en una carpeta `mlruns` al lado de cada notebook, sin servidor: los notebooks también funcionan, pero cada carpeta tiene su propio registro.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué tres cosas hay que versionar en un proyecto de machine learning, y con qué herramienta versiona el autor cada una?**

En un proyecto de machine learning debemos versionar el código, los datos y los modelos. El autor utiliza Git para controlar los cambios en el código, DVC para gestionar las versiones de los datasets y MLflow para registrar los modelos entrenados junto con sus parámetros y métricas.

**¿Por qué no alcanza con Git para versionar los datos?**

Porque los datasets pueden ser muy grandes y cambiar con frecuencia. Git guarda las versiones dentro del historial del repositorio, por lo que mantener múltiples versiones de archivos grandes puede volverlo demasiado pesado. Herramientas como DVC permiten gestionar esas versiones sin almacenar todos los datos directamente en Git.

## Pregunta de apertura

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Necesitaría conocer los hiperparámetros de cada modelo, la semilla utilizada para dividir los datos y cómo se realizó la preparación de las variables. También comprobaría que ambos utilizaron exactamente la misma versión del dataset y calcularon el error con la misma métrica. Con esa información podríamos identificar qué cambió entre las corridas y si los resultados son realmente comparables.

## Ejercicio 1: registrar cada corrida con MLflow

Notebook: [`starter/ejercicio1_registrar_corridas.ipynb`](./starter/ejercicio1_registrar_corridas.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Entrena el bosque y el gradient boosting de la Semana 10, y después la corrida de un compañero. Los 4 `TODO` son el registro. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (antes de los TODO):** sin resolver, el notebook entrena los tres modelos e imprime sus errores. ¿Cuál de los dos gradient boosting tiene razón? ¿Qué parte de la salida te permite saberlo, y qué parte no?

Ambos resultados son válidos para sus respectivas divisiones de datos. El primer gradient boosting obtuvo un MAE de prueba de 1.2750 con la semilla 42, mientras que el del compañero obtuvo 1.1860 con la semilla 7. La salida permite ver que utilizaron semillas diferentes, pero no permite comprobar todavía que los hiperparámetros y la versión del dataset sean los mismos. Por eso, no podemos concluir que uno de los dos modelos sea mejor.

**Respuesta 2 (TODO 2 y 3):** de todo lo que se puede registrar de una corrida, ¿qué crees que explica la diferencia entre los dos gradient boosting? ¿Qué parámetro esperas ver distinto en la tabla del paso 5?

La diferencia parece estar en la división de los datos, porque los dos gradient boosting utilizan el mismo algoritmo, pero fueron evaluados sobre muestras distintas. Espero encontrar una diferencia en `semilla_division`: 42 para el primero y 7 para el compañero. La huella de los datos debería coincidir, ya que ambos utilizan el mismo archivo.

**Respuesta 3 (TODO 4):** el registro ya tiene los parámetros y las métricas. ¿Para qué guardar además el modelo entrenado?

Para poder recuperar el pipeline entrenado y utilizarlo nuevamente sin repetir el entrenamiento. Los parámetros y las métricas nos permiten conocer cómo se ejecutó la corrida y qué resultados obtuvo, pero no contienen los valores que aprendió el modelo. Guardarlo también permite comprobar sus predicciones y reproducir la evaluación sobre los mismos datos.

**Lo que mostró la tabla del paso 5:**

```
run_id tags.mlflow.runName params.semilla_division params.huella_datos params.n_estimators  metrics.mae_prueba  metrics.segundos
d6d688c52b8f4b539c76d08bedad26b1   gradient boosting                       7        4800204d50c0                 100            1.186040          0.164509
c0514bd082144bc787de2f8a69ae59bc   gradient boosting                      42        4800204d50c0                 100            1.274971          0.154516
ef597706900e47819bb42b6f49ad90c7    bosque aleatorio                      42        4800204d50c0                 200            1.299323          0.913639
```

En tabla:
                          
|run_id tags.mlflow.runName| params.semilla_division| params.huella_datos| params.n_estimators|  metrics.mae_prueba|  metrics.segundos|
|--|--|--|--|--|--|
|d6d688c52b8f4b539c76d08bedad26b1|   gradient boosting|                       7 |       4800204d50c0 |                100  |          1.186040 |         0.164509|
|c0514bd082144bc787de2f8a69ae59bc |  gradient boosting   |                   42        |4800204d50c0     |            100     |       1.274971      |    0.154516|
|ef597706900e47819bb42b6f49ad90c7  |  bosque aleatorio   |                   42  |      4800204d50c0           |      200     |       1.299323    |      0.913639|

El gradient boosting del compañero obtuvo un MAE de prueba de 1.186040 con la semilla 7, mientras que el nuestro obtuvo 1.274971 con la semilla 42. El bosque aleatorio obtuvo 1.299323, también con la semilla 42. Las tres corridas tienen la misma huella de datos (`4800204d50c0`), por lo que utilizaron el mismo archivo. La diferencia entre los gradient boosting está en la división de entrenamiento y prueba, no en el dataset.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

NO REALIZADO EN LA CLASE.

**Tu razonamiento:**

NO REALIZADO EN LA CLASE.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

NO REALIZADO EN LA CLASE.

## Ejercicio 2: comparar con evidencia y elegir

Solo o en pareja. Notebook: [`starter/ejercicio2_comparar_y_elegir.ipynb`](./starter/ejercicio2_comparar_y_elegir.ipynb), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1, con los cinco modelos de la Semana 10 en un experimento propio. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

||tags.mlflow.runName|	metrics.mae_entrenamiento|	metrics.mae_prueba|	metrics.segundos|	run_id|
|--|--|--|--|--|--|
|0|	red neuronal|	1.088204|	1.169505|	1.395101|	5d8afdb660214001bb6748089e51435c|
|1|	red neuronal|	1.088204|	1.169505|	1.304000|	b4f89f0a89024ff1af774c804798a146|
|2|	gradient boosting|	1.070539|	1.274971|	0.151662|	787fb6f4b999474393ca80f6599c44ad|
|3|	gradient boosting|	1.070539|	1.274971|	0.167901|	e048ef0e91884b26b0ce1543345c7f39|
|4|	bosque aleatorio|	0.474358|	1.299323|	0.881063|	51fa891268934d119a1e6ec00a1e2493|
|5|	bosque aleatorio|	0.474358|	1.299323|	0.887095|	c261467c7f5e4ef2a45bc2486a207467|
|6|	ridge|	2.493893|	2.556469|	0.012177|	4cbf3f793fed46afab5205ef1321e1a8|
|7|	ridge|	2.493893|	2.556469|	0.014686|	99e5c21b213d4c68a26283304ae96583|
|8|	linea base|	13.199786|	12.530824|	0.024161|	30a6d6e91bf4469983db3c0e7bfbd599|
|9|	linea base|	13.199786|	12.530824|	0.039755|	bd95f9a124f84e5280f18226e7cda8ea|


**Respuesta 1:** ¿qué modelo quedó primero en el registro, y por cuánto le gana al segundo?

El mejor modelo fue la red neuronal, con un MAE de prueba de 1.1695, seguida de Gradient Boosting con 1.2750. La diferencia fue de aproximadamente 0.1055 dólares por viaje a favor de la red neuronal.

**Respuesta 2:** el paso 5 carga el modelo ganador desde el registro y vuelve a calcular su error. ¿Qué te garantiza que coincida con el número registrado? ¿Qué no te garantiza sobre ese modelo?

El modelo recuperado conserva los parámetros aprendidos y las transformaciones del pipeline original. Al evaluarlo sobre los mismos datos de prueba y con la misma métrica, reproduce el MAE registrado de 1.169505. Esto comprueba que el modelo se guardó y recuperó correctamente, pero no garantiza que obtenga el mismo rendimiento con otros datos ni que sea el mejor en cualquier situación.

**Respuesta 3:** si mañana alguien cambia una fila de `viajes_features.csv` y vuelve a entrenar, ¿cómo lo notarías mirando solo el registro?

Compararía la huella del archivo registrada en las corridas. Si alguien modifica una fila de `viajes_features.csv`, su huella debería cambiar, aunque el archivo conserve el mismo nombre. Esto permitiría identificar que las corridas utilizaron versiones diferentes del dataset.

**Si terminas antes:** abre la interfaz de MLflow (pestaña **Ports**, puerto 5050), entra al experimento `tarifa-taxis-comparacion`, selecciona las cinco corridas y compáralas lado a lado con **Compare**. Además del error, ¿qué te dice el registro de la red neuronal que no te diría una tabla escrita a mano?

Además del error, MLflow permite revisar los hiperparámetros de la red neuronal, como sus 64 neuronas ocultas, el optimizador Adam, la tasa de aprendizaje y el límite de 1000 iteraciones. También registra la semilla de división, la huella de los datos y el tiempo de entrenamiento. Esto nos permite conocer las condiciones de la corrida, comparar su costo con otros modelos y recuperar el modelo entrenado.


## Ejercicio 3: el registro de tu proyecto

Usa tu `pipeline_proyecto.py` y los modelos que entrenaste en las dos sesiones de la Semana 10. Esta vez no hay plantilla.

### Entregable 1: `experimentos_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí la preparación y la división y registre en MLflow, en un experimento con el nombre de tu proyecto, una corrida por cada modelo de la Semana 10 (línea base, modelo lineal, ensamblaje, red neuronal). Cada corrida registra:

- Los hiperparámetros del modelo.
- Cómo se dividieron los datos (la semilla, o la fecha de corte si divides por fecha) y la huella del archivo de datos.
- La métrica de tu proyecto en entrenamiento y en prueba, y los segundos de entrenamiento.
- El pipeline entrenado, como artefacto.

Al final, el notebook marca la corrida que eliges como final (`candidato` = `final`), carga su modelo desde el registro y comprueba que reproduce la métrica registrada.

### Entregable 2: `corridas_proyecto.csv`

La tabla de corridas de tu experimento, exportada con `mlflow.search_runs(...).to_csv("corridas_proyecto.csv", index=False)` desde el mismo notebook. Antes de hacer commit, borra las corridas de prueba del experimento de tu proyecto (en la interfaz de MLflow: selecciónalas y **Delete**), reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa.

### Entregable 3: la decisión

**¿Qué modelo elegiste como final? Escribe su `run_id` y los números del registro que sostienen la decisión.**

Elegimos Gradient Boosting como modelo final, con el `run_id` `82c8bc1a272445b79eef00c1d97e0c5d`. Obtuvo la mayor balanced accuracy en prueba, con 0.6975, frente a 0.6796 de la regresión logística y 0.6792 de la red neuronal. También consiguió un F1 de 0.3959, una precision de 0.2801 y un recall de 0.6747, con un tiempo de entrenamiento de 4.10 segundos. Además, recuperamos el modelo desde MLflow y comprobamos que reproduce exactamente la métrica registrada.

**Si otra persona clona tu repositorio, ¿qué necesita para obtener exactamente el mismo número? ¿Qué de eso quedó registrado en MLflow y qué no?**

Necesitaría el mismo dataset, el código del pipeline, las versiones de las librerías y las configuraciones utilizadas para entrenar. MLflow registra la huella SHA-256 del archivo, las fechas de la división temporal, los hiperparámetros, las métricas y el modelo entrenado. Sin embargo, la huella no permite recuperar los datos originales y los artefactos de MLflow están almacenados localmente, fuera del repositorio. Para reproducir exactamente el entrenamiento también sería necesario conservar el entorno y controlar las posibles fuentes de aleatoriedad.

**Recordamos como en anteriores entregas, estas iteraciones del proyecto final no corresponden los modelos ni el estado final del dataset definitivo que se usará en la entrega final. Aun hay columnas por integrar de las tablas del modelo gold y combinaciones de modelos que considerar. Estos resultados corresponden estados exploratorios iniciales que se desarrollaron para acompañar las sesiones de clase.**

## Antes de la próxima sesión

**Lectura:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

Hoy elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

El error podría cambiar si utilizáramos otra división, porque los conjuntos de entrenamiento y prueba tendrían ejemplos diferentes. No podemos saber cuánto variaría sin repetir el experimento con otras semillas. En MLflow registramos la semilla utilizada y el error obtenido, pero no medimos su variación entre distintas divisiones. Para conocerla tendríamos que ejecutar varias corridas y comparar sus resultados.