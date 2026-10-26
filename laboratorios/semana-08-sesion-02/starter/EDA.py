import pandas as pd
import os
import matplotlib.pyplot as plt

# Crear carpeta donde yo pueda guardar todas las imagenes que vamos a generar
# se cree con el programa

os.makedirs('figures', exist_ok=True)

localdir = "starter"

data = pd.read_csv(localdir+'/viajes_taxi.csv')

print("Analisis exploratorio descriptivo")

data.info()

print(data.dtypes)
print(data.isna().sum())
print(data.duplicated().sum())

print(data.describe())

plt.figure()
plt.hist(data['tarifa'], bins=50)
plt.title('Distribution de la tarifa')
plt.xlabel('USD')
plt.ylabel('Numero de viajes')
plt.savefig(localdir+'/figures/hist_tarifa.png')
plt.close()

plt.figure()
plt.boxplot(data['tarifa'])
plt.title('Boxplot de la tarifa')
plt.savefig('./starter/figures/boxplot_tarifa.png')
plt.close()


# Extraccion de valores atipicos

q1 = data['tarifa'].quantile(0.25)
q3 = data['tarifa'].quantile(0.75)

iqr = q3-q1
lim_sup = q3 + 1.5 * iqr
lim_inf = q1 - 1.5 * iqr


print("atipicos")
atipicos = data.query(f'tarifa < {lim_inf} or tarifa > {lim_sup}')
print(atipicos)



plt.figure()
plt.scatter(data['tarifa'], data['distancia_km'])
plt.title('Dispercion de la tarifa vs distancia_km')
plt.savefig('./starter/figures/scatter_distancia_vs_tarifa.png')
plt.close()


sin_atipicos = data.query(f'tarifa >= {lim_inf} and tarifa <= {lim_sup}')
plt.figure()
plt.scatter(sin_atipicos['tarifa'], sin_atipicos['distancia_km'])
plt.title('Dispercion de la tarifa vs distancia_km')
plt.savefig('./starter/figures/filtrado_scatter_distancia_vs_tarifa.png')
plt.close()

print(sin_atipicos.corr(numeric_only=True))

