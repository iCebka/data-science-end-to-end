# Semana 9, Sesión 1: Ingeniería de características: principios aplicados

> Objetivo de la sesión: explicar por qué una misma variable cruda puede necesitar codificaciones distintas según el modelo que la consuma, y diseñar al menos dos features derivadas para el dataset de tu proyecto.

**Lectura previa de esta sesión:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen (O'Reilly, 2022), Capítulo 5: Feature Engineering.

## Requisitos antes de esta sesión

pandas y matplotlib, de la sesión pasada. numpy se instala junto con pandas; verifica con:

```
python3 -c "import numpy; print(numpy.__version__)"
```

Ten a mano tu notebook de exploración de la Semana 8, Sesión 2: las features de hoy salen de tus 3 hallazgos.

Los laboratorios de hoy siguen siendo notebooks: diseñar una feature es probar varias ideas y mirar cuál aporta. La próxima sesión, las features que se queden pasan a un script, que es lo que corre en producción.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**

Una feature tiene sentido de negocio cuando representa un comportamiento o una relación que podemos interpretar dentro del problema que queremos resolver. Mejorar una métrica de predicción no necesariamente significa que la variable aporte información útil o comprensible. El artículo propone construir features a partir de proporciones, tendencias y distintos niveles de granularidad, de modo que podamos explicar qué información contienen y por qué deberían influir en la predicción.

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

El artículo presenta el caso de las transacciones en línea para medir qué tanto utiliza un cliente los canales digitales. Contar únicamente las transacciones puede ser engañoso: dos clientes pueden haber realizado cinco compras en línea, pero si uno hizo 100 compras en total y el otro solo cinco, sus comportamientos son distintos. Al dividir las transacciones en línea para el total de transacciones obtenemos una nueva columna con proporciones del 5 % y 100 %, respectivamente. Esta feature representa mejor la preferencia de cada cliente por el canal digital.

## Pregunta de apertura

**La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?**

Podemos construir dos variables derivadas de las columnas existentes. La primera, es_tarifa_fija, identifica los viajes entre JFK y Manhattan mediante una variable binaria. La segunda, es_hora_pico, se obtiene de la hora de inicio y señala los viajes realizados en horarios de mayor congestión. Así podemos incorporar al modelo dos condiciones que podrían afectar la tarifa y que no están representadas directamente en las variables originales.
## Ejercicio 1: features de fecha y hora

Notebook: [`starter/ejercicio1_features_tiempo.ipynb`](./starter/ejercicio1_features_tiempo.ipynb), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada. Ya quita los errores confirmados, convierte `fecha_hora_inicio` y compara la tarifa por kilómetro según la hora pico; faltan 4 `TODO`: hora, día de la semana, fin de semana, hora pico y la hora como ciclo. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 4), antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?


Si representamos la hora como un número entre 0 y 23, el modelo consideraría que entre las 23:00 y las 00:00 hay una distancia de 23 unidades. Sin embargo, en el reloj estas horas están separadas por una sola hora. Esto ocurre porque la representación numérica no considera que el tiempo es cíclico, por lo que podemos utilizar seno y coseno para representar correctamente esa continuidad.


**Respuesta 2 (TODO 3):** sin el TODO 3, el Paso 4 muestra un solo grupo. Con la hora pico construida, ¿esperas que la tarifa por kilómetro sea más alta o más baja en hora pico? ¿Por qué, si la tarifa depende de la distancia y de la duración?
Respuesta 3

Esperaría que la tarifa por kilómetro sea más alta durante las horas pico, porque la congestión puede aumentar la duración del viaje incluso cuando la distancia recorrida es la misma. Si la tarifa depende tanto del tiempo como de la distancia, un viaje más lento puede terminar costando más por kilómetro.

**Respuesta 3:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

La variable es_hora_pico puede aportar información indirecta sobre la congestión y los posibles retrasos durante el recorrido. Aunque no permite conocer la duración exacta de un viaje, puede ayudar al modelo a estimar si es probable que tome más tiempo de lo habitual para una determinada distancia.

**Lo que confirmó la salida, TODO por TODO:**

```python
"""Ejercicio 1: features de fecha y hora de viajes_taxi.csv.

Corre el script antes de tocar nada:

    python3 ejercicio1_features_tiempo.py

Corre sin errores y sin los TODO resueltos.
"""

import numpy as np
import pandas as pd

viajes = pd.read_csv("viajes_taxi.csv")
viajes = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)].copy()
inicio = pd.to_datetime(viajes["fecha_hora_inicio"])

print("Paso 1: partes de la fecha")
# TODO 1: la hora (0 a 23) y el dia de la semana (0 es lunes) de inicio.
viajes["hora"] = inicio.dt.hour
viajes["dia_semana"] = inicio.dt.day_of_week
# TODO 2: es_fin_de_semana, 1 los sabados y domingos, 0 el resto.
viajes["es_fin_de_semana"] = (viajes["dia_semana"] >= 5).astype(int)

print("Paso 2: hora pico")
horas_pico = [7, 8, 9, 16, 17, 18, 19]
# TODO 3: es_hora_pico, 1 si la hora esta en horas_pico y no es fin de
# semana, 0 el resto.
viajes["es_hora_pico"] = (
    viajes["hora"].isin(horas_pico)
    & (viajes["es_fin_de_semana"] == 0)
).astype(int)

print("Paso 3: la hora como ciclo")
# TODO 4: hora_sin y hora_cos, la hora como un punto en un reloj de 24
# horas.
viajes["hora_sin"] = np.sin( 2 * np.pi * viajes["hora"] / 24)
viajes["hora_cos"] = np.cos( 2 * np.pi * viajes["hora"] / 24)

columnas = ["fecha_hora_inicio", "hora", "dia_semana", "es_fin_de_semana", "es_hora_pico", "hora_sin", "hora_cos"]
print(viajes[columnas].head())

print()
print("Paso 4: la hora pico aporta algo a la tarifa?")
normales = viajes[(viajes["tarifa"] != 70.0) & (viajes["distancia_km"] > 0)].copy()
normales["tarifa_por_km"] = normales["tarifa"] / normales["distancia_km"]
print("Tarifa por km, mediana, segun hora pico:")
print(normales.groupby("es_hora_pico")["tarifa_por_km"].median().round(2))

print()
print("Paso 5: que tan lejos quedan dos horas")


def distancia(fila_a, fila_b):
    a = np.array([fila_a["hora_sin"], fila_a["hora_cos"]])
    b = np.array([fila_b["hora_sin"], fila_b["hora_cos"]])
    return np.sqrt(((a - b) ** 2).sum())


ejemplo = viajes.drop_duplicates("hora").set_index("hora")
for h1, h2 in [(23, 0), (11, 12)]:
    if h1 in ejemplo.index and h2 in ejemplo.index:
        print(f"{h1} y {h2}: como numero {abs(h1 - h2)}, como ciclo {distancia(ejemplo.loc[h1], ejemplo.loc[h2]):.2f}")
    else:
        print(f"{h1} y {h2}: todavia no hay horas distintas para comparar")

```

```
Attaching to s9w1e1
s9w1e1  | Paso 1: partes de la fecha
s9w1e1  | Paso 2: hora pico
s9w1e1  | Paso 3: la hora como ciclo
s9w1e1  |   fecha_hora_inicio  hora  ...  hora_sin      hora_cos
s9w1e1  | 0  2026-09-06 13:46    13  ... -0.258819 -9.659258e-01
s9w1e1  | 1  2026-10-01 22:22    22  ... -0.500000  8.660254e-01
s9w1e1  | 2  2026-09-17 19:37    19  ... -0.965926  2.588190e-01
s9w1e1  | 3  2026-10-04 18:09    18  ... -1.000000 -1.836970e-16
s9w1e1  | 4  2026-10-12 09:28     9  ...  0.707107 -7.071068e-01
s9w1e1  | 
s9w1e1  | [5 rows x 7 columns]
s9w1e1  | 
s9w1e1  | Paso 4: la hora pico aporta algo a la tarifa?
s9w1e1  | Tarifa por km, mediana, segun hora pico:
s9w1e1  | es_hora_pico
s9w1e1  | 0    3.22
s9w1e1  | 1    3.52
s9w1e1  | Name: tarifa_por_km, dtype: float64
s9w1e1  | 
s9w1e1  | Paso 5: que tan lejos quedan dos horas
s9w1e1  | 23 y 0: como numero 23, como ciclo 0.26
s9w1e1  | 11 y 12: como numero 1, como ciclo 0.26
s9w1e1 exited with code 0
```

Se obtuvieron correctamente la hora, el día de la semana y las variables que identifican fines de semana y horas pico. La mediana de la tarifa por kilómetro fue de $3.22 fuera de hora pico y $3.52 durante hora pico, lo que confirma que los viajes tienden a costar más por kilómetro en esos horarios.

En el Paso 5, la distancia entre las 23:00 y las 00:00 pasó de 23 unidades a 0.26 con la representación cíclica, mientras que entre las 11:00 y las 12:00 se mantuvo la separación numérica de 1 unidad y se obtuvo también 0.26 en la representación cíclica. Esto confirma que la codificación con seno y coseno representa correctamente la cercanía entre horas consecutivas.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

NO REALIZADO EN CLASE

**Tu razonamiento:**

NO REALIZADO EN CLASE

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

NO REALIZADO EN CLASE

## Ejercicio 2: zona, codificación y disponibilidad

Solo o en pareja. Notebook: [`starter/ejercicio2_features_zona.ipynb`](./starter/ejercicio2_features_zona.ipynb), sobre el mismo `viajes_taxi.csv`. Completa los 4 `TODO`. La última celda imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.


```
s9w1e2  | Viajes con tarifa fija: 300
s9w1e2  | Tarifa minima: 70.0
s9w1e2  | Tarifa maxima: 70.0
s9w1e2  | Columnas one-hot: ['origen_Brooklyn', 'origen_Harlem', 'origen_JFK Airport', 'origen_Lower Manhattan', 'origen_Midtown', 'origen_Queens', 'origen_Upper East Side', 'origen_Upper West Side']
s9w1e2  |         zona_origen  origen_ordinal
s9w1e2  | 0          Brooklyn               0
s9w1e2  | 3            Harlem               1
s9w1e2  | 52      JFK Airport               2
s9w1e2  | 1   Lower Manhattan               3
s9w1e2  | 5           Midtown               4
s9w1e2  | 13           Queens               5
s9w1e2  | 2   Upper East Side               6
s9w1e2  | 6   Upper West Side               7
s9w1e2  | id_viaje: False
s9w1e2  | fecha_hora_inicio: True
s9w1e2  | zona_origen: True
s9w1e2  | zona_destino: True
s9w1e2  | distancia_km: True
s9w1e2  | duracion_min: False
s9w1e2  | pasajeros: True
s9w1e2  | metodo_pago: False
s9w1e2  | tarifa: False
s9w1e2  | propina: False
s9w1e2  | 
s9w1e2  | Comprobacion
s9w1e2  | TODO 1: OK
s9w1e2  | TODO 2: OK
s9w1e2  | TODO 3: OK
s9w1e2  | TODO 4: OK
s9w1e2 exited with code 0
```

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?

Se identificaron 300 viajes con es_tarifa_fija igual a 1, todos con una tarifa de $70, tanto mínima como máxima. Esto muestra que existe una relación directa entre determinadas rutas y el precio del viaje. Un modelo lineal puede aprovechar esta columna para aprender un ajuste específico de la tarifa para esos recorridos, en lugar de depender únicamente de la distancia y otras variables.

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?

La codificación one-hot genera 8 columnas, una por cada zona de origen, mientras que la ordinal genera una sola columna con valores entre 0 y 7. Para una regresión lineal utilizaría one-hot, porque permite identificar cada zona sin establecer relaciones numéricas artificiales. La codificación ordinal introduce un orden y distancias entre zonas que no necesariamente tienen significado, haciendo que el modelo interprete diferencias entre códigos como diferencias proporcionales en su efecto sobre la tarifa.

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?

Al momento de pedir el taxi todavía no conocemos la duración real del viaje, la tarifa final ni la propina. También consideramos que el método de pago aún no se ha definido. Si entrenamos el modelo con estas variables, podría aprender relaciones utilizando información que no estará disponible al realizar la predicción. Esto produciría una evaluación artificialmente favorable durante el entrenamiento y dificultaría su funcionamiento en condiciones reales, ya lo conocemos, es data leakage.

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de tu exploración de la Semana 8 y copia la plantilla [`plantilla-features/features_proyecto.ipynb`](./plantilla-features/features_proyecto.ipynb) a `psets/proyecto/modelado/`, al lado de tu notebook de exploración. Todo lo que pide este ejercicio queda en el notebook: el código de cada feature, la evidencia y la justificación en las celdas de texto.

### Paso 1: el momento de la predicción (5 minutos)

Escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

Nuestro modelo busca predecir si el valor de mercado de un futbolista disminuirá al menos un 40% aproximadamente 12 meses después de una valoración determinada.

La predicción se realiza en la fecha de esa valoración. En ese momento conocemos la edad, posición, altura, pie dominante y valor de mercado actual del jugador. También podemos utilizar sus valoraciones anteriores y las estadísticas deportivas registradas hasta esa fecha.

No conocemos el valor que tendrá el futbolista dentro de 12 meses. Esa información se utiliza únicamente para construir `TARGET_DEPRECIACION_40` durante la preparación del dataset, pero no forma parte de las entradas del modelo. La idea es predecir una potencial depreciación de un 40% recibiendo el estado del jugador en este momento, capturado por la valuación en este momento y datos de juego hasta este momento, en entrenamiento se tiene un set de estados capturados (valuaciones y datos deportivos) cuyo futuro depreciado o no depreciado ya es conocido, esa es la label que vamos a predecir.

### Paso 2: construye las 2 features (15 minutos)

Completa la plantilla con 2 columnas nuevas, derivadas de las existentes. Al menos una tiene que salir de uno de tus hallazgos de la exploración.

A partir de los hallazgos de la exploración de la Semana 8, creamos dos características binarias: ES_MAYOR_30, que identifica a los jugadores de 30 años o más, y CAIDA_PREVIA_20, que identifica a quienes habían perdido al menos un 20% de su valor respecto a una valoración histórica anterior.

```python
A partir de los hallazgos de la exploración de la Semana 8, creamos dos características binarias: ES_MAYOR_30, que identifica a los jugadores de 30 años o más, y CAIDA_PREVIA_20, que identifica a quienes habían perdido al menos un 20% de su valor respecto a una valoración histórica anterior.
```

Para la segunda característica conservamos los valores faltantes, porque no tener una valoración histórica disponible no significa que el jugador no haya sufrido una caída de valor.

El dataset resultante conserva las 450.078 observaciones y pasa de 24 a 26 columnas.

### Paso 3: justifica cada feature (5 minutos)

Debajo de cada feature: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta, y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar. Antes de hacer commit, reinicia el kernel y corre todo.

**Feature 1: ES_MAYOR_30**

En la exploración de la Semana 8 encontramos que la frecuencia de depreciaciones aumenta considerablemente con la edad, especialmente a partir de los 30 años. Por ello, creamos una variable binaria que identifica a los jugadores que han alcanzado esa edad.

Los resultados muestran que, de las 381.866 observaciones correspondientes a jugadores menores de 30 años, el 10,0% presenta una depreciación futura de al menos el 40%. En cambio, entre las 68.212 observaciones de jugadores de 30 años o más, el porcentaje aumenta al 34,5%.

Esta diferencia sugiere que la nueva característica podría aportar información al modelo. Además, la edad está disponible en el momento de la predicción, por lo que no introduce información futura. Al ser una variable binaria, puede utilizarse directamente tanto en modelos lineales como en modelos basados en árboles, sin necesidad de codificación one-hot.

**Feature 2: CAIDA_PREVIA_20**

Durante el EDA también encontramos una relación entre la evolución histórica del valor de mercado y la depreciación futura. Los jugadores que ya habían perdido una parte importante de su valor presentaban una mayor frecuencia de nuevas depreciaciones.

Para representar este comportamiento, creamos una variable binaria que indica si el jugador había perdido al menos un 20% de su valor respecto a una valoración histórica cercana a 12 meses antes.

Los resultados muestran tres grupos:

- Sin caída previa de al menos el 20%: 268.774 observaciones, con una tasa de depreciación futura del 11,5%.

- Con caída previa de al menos el 20%: 87.195 observaciones, con una tasa del 27,8%.

- Sin información histórica suficiente: 94.109 observaciones, con una tasa del 6,9%.

La diferencia entre los grupos sugiere que la evolución previa del valor puede aportar información predictiva. También observamos que las filas sin historial suficiente presentan un comportamiento diferente, por lo que no debemos confundir los valores faltantes con la ausencia de depreciación previa.

Esta característica utiliza únicamente información anterior a la fecha de predicción, por lo que no introduce fuga de información. Es binaria y no requiere codificación one-hot. Por ahora conservamos sus valores faltantes y dejaremos su tratamiento para la etapa de preprocesamiento.


```
Attaching to s9w1e3
s9w1e3  | Depreciacion segun edad:
s9w1e3  |               count   mean
s9w1e3  | ES_MAYOR_30               
s9w1e3  | 0            381866  0.100
s9w1e3  | 1             68212  0.345
s9w1e3  | 
s9w1e3  | Depreciacion segun caida previa:
s9w1e3  |                   count   mean
s9w1e3  | CAIDA_PREVIA_20               
s9w1e3  | 0                268774  0.115
s9w1e3  | 1                 87195  0.278
s9w1e3  | <NA>              94109  0.069
s9w1e3  | 
s9w1e3  | Primeras filas:
s9w1e3  |    PLAYER_ID VALUATION_DATE  ...  ES_MAYOR_30  CAIDA_PREVIA_20
s9w1e3  | 0         10     2004-10-04  ...            0             <NA>
s9w1e3  | 1         26     2004-10-04  ...            0             <NA>
s9w1e3  | 2         65     2004-10-04  ...            0             <NA>
s9w1e3  | 3         77     2004-10-04  ...            0             <NA>
s9w1e3  | 4        109     2004-10-04  ...            0             <NA>
s9w1e3  | 
s9w1e3  | [5 rows x 26 columns]
s9w1e3 exited with code 0
```

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

La media y la desviación estándar deben calcularse **únicamente con los viajes del conjunto de entrenamiento**. Después utilizamos esos mismos valores para transformar tanto los datos de entrenamiento como los de prueba.

Si calculamos estas estadísticas utilizando todos los viajes, estaríamos incorporando información del conjunto de prueba durante el preprocesamiento. Esto produciría una fuga de información (*data leakage*) y podría generar una evaluación demasiado optimista del modelo.

Por tanto, primero debemos dividir los datos en entrenamiento y prueba, luego ajustar el escalador con los datos de entrenamiento y finalmente aplicar esa transformación a ambos conjuntos.