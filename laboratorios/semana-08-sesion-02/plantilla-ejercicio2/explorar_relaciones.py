import os

import matplotlib.pyplot as plt
import pandas as pd

# Ejecutar desde plantilla-ejercicio2/ (o mediante Docker Compose).
os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("../starter/viajes_taxi.csv")

print("=== Ejercicio 2: relaciones entre variables ===")
print(f"Viajes originales: {len(viajes)}")

# TODO 1: quitar solo los errores confirmados en el Ejercicio 1.
# No eliminamos las demas observaciones marcadas como atipicas por IQR.
error_tarifa = viajes["tarifa"].eq(999.99)
error_duracion = viajes["duracion_min"].lt(0)
limpios = viajes.loc[~(error_tarifa | error_duracion)].copy()

print(f"Tarifas 999.99: {error_tarifa.sum()}")
print(f"Duraciones negativas: {error_duracion.sum()}")
print(f"Viajes que quedan: {len(limpios)}")

# TODO 2: dispersion de distancia (x) contra tarifa (y).
plt.figure(figsize=(8, 5))
plt.scatter(limpios["distancia_km"], limpios["tarifa"], alpha=0.4, s=14)
plt.title("Distancia vs. tarifa")
plt.xlabel("Distancia (km)")
plt.ylabel("Tarifa (USD)")
plt.tight_layout()
plt.savefig("figuras/distancia_vs_tarifa.png", dpi=150)
plt.close()

# Contamos los puntos en la banda horizontal de tarifa = 70 USD.
viajes_70 = limpios[limpios["tarifa"].eq(70)]
print("\n--- Distancia contra tarifa ---")
print(f"Viajes con tarifa exactamente 70 USD: {len(viajes_70)}")
if not viajes_70.empty:
    print(
        "Distancia de esos viajes: "
        f"{viajes_70['distancia_km'].min():.2f} a "
        f"{viajes_70['distancia_km'].max():.2f} km"
    )
print(f"Correlacion distancia-tarifa: {limpios['distancia_km'].corr(limpios['tarifa']):.4f}")

# TODO 3: propina promedio y cantidad de propinas en 0 por metodo de pago.
# dropna=False permite mostrar tambien los viajes sin metodo registrado.
propinas_por_pago = (
    limpios.groupby("metodo_pago", dropna=False)
    .agg(
        viajes=("propina", "size"),
        propina_promedio=("propina", "mean"),
        propinas_cero=("propina", lambda valores: valores.eq(0).sum()),
    )
)
propinas_por_pago["porcentaje_cero"] = (
    100 * propinas_por_pago["propinas_cero"] / propinas_por_pago["viajes"]
)
print("\n--- Propina por metodo de pago ---")
print(propinas_por_pago.round(2).to_string())

# TODO 4: correlaciones de las cuatro variables con la tarifa.
columnas = ["distancia_km", "duracion_min", "pasajeros", "propina", "tarifa"]
correlaciones = limpios[columnas].corr()["tarifa"].drop("tarifa")
print("\n--- Correlaciones con la tarifa ---")
print(correlaciones.round(4).to_string())

# TODO 5: pasajeros vacios y pasajeros en 0 (no son equivalentes).
pasajeros_vacios = limpios["pasajeros"].isna().sum()
pasajeros_cero = limpios["pasajeros"].eq(0).sum()
print("\n--- Calidad de la variable pasajeros ---")
print(f"Viajes con pasajeros vacio: {pasajeros_vacios}")
print(f"Viajes con pasajeros en 0: {pasajeros_cero}")
print("\nGrafico guardado en figuras/distancia_vs_tarifa.png")
