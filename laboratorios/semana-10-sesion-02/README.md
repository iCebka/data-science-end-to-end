# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

El aprendizaje automático clásico conviene cuando tenemos datos estructurados y buscamos resolver problemas específicos, como clasificación, regresión o predicción de valores. Los LLM son más adecuados para trabajar con lenguaje natural, especialmente cuando necesitamos interpretar, resumir o generar texto.

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

Para nuestro proyecto considero que pesa más la explicabilidad. Queremos estimar el riesgo de depreciación de los jugadores, y estas predicciones podrían influir en decisiones económicas importantes para un club. Por eso, además de obtener una buena predicción, necesitamos poder identificar qué variables influyeron en ella y ofrecer información que ayude a justificar las decisiones.

## Pregunta de apertura

**Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?**

Además del error, mediría el tiempo de entrenamiento y de respuesta, el costo computacional de cada opción y su capacidad para explicar las predicciones. También consideraría la consistencia de los resultados, ya que una misma entrada debería producir la misma tarifa, y la facilidad de integrar y mantener cada modelo dentro de la aplicación.

## Ejercicio 1: una red neuronal frente al ensamblaje

Archivo: [`starter/ejercicio1_red_neuronal.py`](./starter/ejercicio1_red_neuronal.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). El script ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división. Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-02/starter
python3 ejercicio1_red_neuronal.py
```

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

La advertencia ConvergenceWarning indica que la red neuronal alcanzó el límite de iteraciones sin cumplir el criterio de convergencia. Al aumentar el límite a 1000 iteraciones, esperaría que pueda ajustar mejor sus parámetros y reducir el error, aunque esto no garantiza que supere al ensamblaje y seguramente aumentará el tiempo de entrenamiento.

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

La red tiene que aprender los pesos de las 25 entradas hacia las 64 neuronas, los 64 sesgos de la capa oculta, los 64 pesos hacia la salida y un sesgo final.

$$(25 \times 64) + 64 + 64 + 1 = 1729$$

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

Esperaría que la distancia del viaje sea la variable más importante, porque normalmente explica buena parte de la tarifa. Entre los dos modelos, el ensamblaje permitiría justificar mejor la estimación, ya que podemos consultar la importancia de sus variables. Sin embargo, para explicar por qué un viaje específico cuesta 70 dólares necesitaríamos analizar esa predicción individual, no solamente las importancias generales.

**Lo que confirmó la salida, TODO por TODO:**

```
s10e1  | gradient boosting    entrenamiento  1.0705   prueba  1.2750     0.16 s
s10e1  | red neuronal         entrenamiento  1.0882   prueba  1.1695     1.39 s
s10e1  | 
s10e1  |            modelo  mae_entrenamiento  mae_prueba  segundos
s10e1  | gradient boosting             1.0705      1.2750     0.164
s10e1  |      red neuronal             1.0882      1.1695     1.394
s10e1  | 
s10e1  | Parametros de la red: 1729
s10e1  | Las 3 columnas mas importantes para el ensamblaje:
s10e1  | numericas__distancia_km      0.9824
s10e1  | ya_listas__es_tarifa_fija    0.0120
s10e1  | ya_listas__es_hora_pico      0.0028

```

La red neuronal consiguió un menor error de prueba que el ensamblaje, aunque necesitó más tiempo de entrenamiento. También se confirmó que la distancia es, por mucho, la variable más importante para el ensamblaje, con una importancia de 0.9824.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

ACTIVIDAD NO REALIZADA EN CLASE.

**Tu razonamiento:**

ACTIVIDAD NO REALIZADA EN CLASE.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

ACTIVIDAD NO REALIZADA EN CLASE.

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Archivo: [`starter/ejercicio2_tres_soluciones.py`](./starter/ejercicio2_tres_soluciones.py), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre el script. Además de entrenar, escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y vuelve a correr el script hasta que todas las comprobaciones digan `OK`. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

```
python3 ejercicio2_tres_soluciones.py
```

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

Se utilizó ChatGPT como LLM, el modelo fue GPT-Sol6.1. Se probaron 3 intentos con el mismo prompt.

| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | 0.8957 | 0.15s | No medido pero se estima bajo porque es inferencia local | Sí, se analiza las variables del modelo |
| Red neuronal | 0.9637 | 1.35s | No medido pero se estima bajo porque es inferencia local | Dificil de interpretar, pero parámetros controlables |
| LLM | 3.8515 | No requiere entrenamiento, alrededor de 1 minuto para generar respuesta | $0.7255 | No es posible, distintos intentos generan distintas respuestas sin ningun cambio en como se ejecuta la inferencia |

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

No, las tres ejecuciones dieron resultados diferentes. El MAE del LLM pasó de 3.8515 en la primera a 1.0125 en la segunda y 3.7175 en la tercera. Incluso hubo un viaje para el que estimó $7.02 en una ejecución y $70 en las otras dos.

Esto sería un problema para una aplicación de taxis, porque un mismo viaje podría recibir tarifas muy diferentes sin que cambien sus características. Necesitamos que las estimaciones sean consistentes para evitar cobros arbitrarios.

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

La peor predicción ocurrió en el viaje de la posición 12 de la muestra. Su tarifa real era de $11.46, pero el LLM estimó $70 en dos ejecuciones, produciendo un error absoluto de $58.54. En la segunda ejecución estimó $7.02, mucho más cerca del valor real.

No tenemos todavía las características completas de ese viaje para identificar qué lo hace particular. Sin embargo, el ensamblaje tuvo acceso a 2380 ejemplos de entrenamiento, por lo que pudo aprender relaciones entre las características de los viajes y sus tarifas con mucha más información que los ejemplos proporcionados al LLM. Esto no garantiza que prediga correctamente ese caso, pero sus resultados muestran un error promedio considerablemente menor.

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

El LLM no se evaluó sobre los 595 viajes porque habría que proporcionarle muchas más entradas, posiblemente en varios prompts, aumentando el costo y el trabajo necesario. En cambio, los modelos tradicionales pueden procesar todo el conjunto de prueba directamente.

Una comparación sobre apenas 20 viajes es poco confiable, porque unos pocos casos pueden alterar considerablemente el promedio. De hecho, observamos que una sola predicción del LLM cambió mucho el MAE entre ejecuciones. Para obtener conclusiones más sólidas necesitaríamos evaluar una muestra mayor y repetir las pruebas.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.py`

Un script que entrene, con tu pipeline y tu división, el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

_(tu tabla)_

_(tu decisión)_

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

_(tu respuesta)_
