import numpy as np
import pandas as pd
import os

# Parámetros del experimento
CSV_PATH = 'our_dataset/fact_player_valuation.csv'
HORIZONTE_MESES = 12
TOLERANCIA_DIAS = 65
OUTPUT_PATH = f'output/valuaciones_h{HORIZONTE_MESES}m_tol{TOLERANCIA_DIAS}d.csv'


def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    # La tabla está ordenada por PLAYER_ID y luego por fecha.
    fechas = df['VALUATION_DATE'].to_numpy(dtype='datetime64[ns]')
    metas = df['FECHA_OBJETIVO'].to_numpy(dtype='datetime64[ns]')
    valores = df['MARKET_VALUE_IN_EUR'].to_numpy(dtype=float)
    jugadores = df['PLAYER_ID'].to_numpy()
    n = len(df)

    conteos = np.zeros(n, dtype=np.int32)
    futura_fecha = np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]')
    futura_valor = np.full(n, np.nan)
    desviacion = np.full(n, np.nan)

    limites = np.r_[0, np.flatnonzero(jugadores[1:] != jugadores[:-1]) + 1, n]
    for comienzo, final in zip(limites[:-1], limites[1:]):
        t = fechas[comienzo:final]
        m = metas[comienzo:final]
        v = valores[comienzo:final]
        k = len(t)
        inicio = np.searchsorted(t, t, side='right')
        fin = np.searchsorted(t, m, side='right')
        conteos[comienzo:final] = fin - inicio

        pos = np.searchsorted(t, m, side='left')
        mejor = np.full(k, -1, dtype=int)
        distancia_mejor = np.full(k, np.inf)
        for candidato in (pos - 1, pos):
            valido = (candidato >= inicio) & (candidato < k)
            filas = np.flatnonzero(valido)
            indices = candidato[filas]
            distancias = np.abs((t[indices] - m[filas]) / np.timedelta64(1, 'D'))
            aceptar = (distancias <= TOLERANCIA_DIAS) & (distancias <= distancia_mejor[filas])
            elegidas = filas[aceptar]
            mejor[elegidas] = indices[aceptar]
            distancia_mejor[elegidas] = distancias[aceptar]

        encontrada = np.flatnonzero(mejor >= 0)
        globales = comienzo + encontrada
        futuras = mejor[encontrada]
        futura_fecha[globales] = t[futuras]
        futura_valor[globales] = v[futuras]
        desviacion[globales] = (t[futuras] - m[encontrada]) / np.timedelta64(1, 'D')

    proporcion = np.divide(
        futura_valor, valores,
        out=np.full(n, np.nan),
        where=(valores > 0) & np.isfinite(futura_valor)
    )
    df['N_VALUACIONES_HORIZONTE'] = conteos
    df['FECHA_VALUACION_FUTURA'] = futura_fecha
    df['DIAS_DESVIACION'] = desviacion
    df['VALOR_FUTURO_EUR'] = futura_valor
    df['PROPORCION_VALOR'] = proporcion
    df['VARIACION_PCT'] = (proporcion - 1) * 100
    return df


def main():


    os.makedirs("output", exist_ok=True)

    if HORIZONTE_MESES <= 0 or TOLERANCIA_DIAS < 0:
        raise ValueError('HORIZONTE_MESES debe ser positivo y TOLERANCIA_DIAS no negativo')

    df = pd.read_csv(CSV_PATH, parse_dates=['VALUATION_DATE'])
    columnas = ['PLAYER_ID', 'VALUATION_DATE', 'MARKET_VALUE_IN_EUR']
    df = df.dropna(subset=columnas).copy()
    df = df.sort_values(['PLAYER_ID', 'VALUATION_DATE'])
    if df.duplicated(['PLAYER_ID', 'VALUATION_DATE']).any():
        raise ValueError('La tabla Gold viola la unicidad (PLAYER_ID, VALUATION_DATE)')
    df['FECHA_OBJETIVO'] = df['VALUATION_DATE'] + pd.DateOffset(months=HORIZONTE_MESES)

    resultado = calcular_metricas(df)
    resultado.to_csv(OUTPUT_PATH, index=False)

    print(f'Filas de entrada y salida: {len(resultado):,}')
    print(f'Jugadores: {resultado.PLAYER_ID.nunique():,}')
    print(f'Con alguna valoración dentro del horizonte: {(resultado.N_VALUACIONES_HORIZONTE > 0).sum():,}')
    print(f'Con valoración cercana al objetivo (±{TOLERANCIA_DIAS} días): {resultado.VALOR_FUTURO_EUR.notna().sum():,}')
    print(f'Con proporción calculable: {resultado.PROPORCION_VALOR.notna().sum():,}')
    print(f'Archivo generado: {OUTPUT_PATH}')
    print(resultado[['PLAYER_ID', 'VALUATION_DATE', 'MARKET_VALUE_IN_EUR',
                     'N_VALUACIONES_HORIZONTE', 'FECHA_VALUACION_FUTURA',
                     'VALOR_FUTURO_EUR', 'PROPORCION_VALOR', 'VARIACION_PCT']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
