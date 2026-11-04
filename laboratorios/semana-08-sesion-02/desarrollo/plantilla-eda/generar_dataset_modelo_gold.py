"""Enriquece valoraciones etiquetadas con atributos de Gold conocidos antes del corte.

Uso:
 python generar_dataset_modelo_gold.py \
   --etiquetas valuaciones_etiquetadas_u40.csv \
   --valoraciones fact_player_valuation.csv \
   --jugadores dim_player.csv \
   --partidos fact_player_match.csv \
   --salida output/modelo

Genera:
  auditoria_enriquecida.csv  (también con campos futuros, para inspección)
  modelo_etiquetado.csv     (solo inputs elegibles + target)
  resumen.txt

La fecha base es VALUATION_DATE, no FECHA_VALUACION_FUTURA.
Las ventanas deportivas son [fecha_corte - 365d, fecha_corte), y 180d análogamente.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def leer_csv(path, obligatorias):
    d = pd.read_csv(path, low_memory=False)
    d.columns = d.columns.str.strip().str.upper()
    falta = sorted(set(obligatorias).difference(d.columns))
    if falta:
        raise ValueError(f'{path}: faltan columnas {falta}. Presentes: {d.columns.tolist()}')
    return d


def asof_por_jugador(base, historia, fecha_izq, fecha_der, direccion='backward', tolerancia=None):
    # merge_asof requiere ordenar por la fecha globalmente (no primero por jugador).
    base = base.sort_values([fecha_izq, 'PLAYER_ID'], kind='stable')
    historia = historia.sort_values([fecha_der, 'PLAYER_ID'], kind='stable')
    return pd.merge_asof(base, historia, by='PLAYER_ID',
                         left_on=fecha_izq, right_on=fecha_der,
                         direction=direccion, tolerance=tolerancia)


def enriquecer_valoraciones(base, valoraciones, tolerancia_previa):
    v = leer_csv(valoraciones, ['PLAYER_ID','VALUATION_DATE','MARKET_VALUE_IN_EUR'])
    v['VALUATION_DATE'] = pd.to_datetime(v['VALUATION_DATE'], errors='coerce')
    v['PLAYER_ID'] = pd.to_numeric(v['PLAYER_ID'], errors='coerce')
    v = v.dropna(subset=['PLAYER_ID', 'VALUATION_DATE'])
    if v.duplicated(['PLAYER_ID', 'VALUATION_DATE']).any():
        raise ValueError('La fact de valoraciones debe tener unicidad (jugador, fecha).')
    h = v[['PLAYER_ID','VALUATION_DATE','MARKET_VALUE_IN_EUR']].rename(columns={
        'VALUATION_DATE': 'FECHA_VALORACION_PREVIA',
        'MARKET_VALUE_IN_EUR': 'VALOR_PREVIO_12M_EUR'})
    base['FECHA_OBJETIVO_PREVIA'] = base['VALUATION_DATE'] - pd.DateOffset(months=12)
    base = asof_por_jugador(base, h, 'FECHA_OBJETIVO_PREVIA',
                           'FECHA_VALORACION_PREVIA', 'nearest',
                           pd.Timedelta(days=tolerancia_previa))
    base['VARIACION_PREVIA_12M_PCT'] = np.where(
        base['VALOR_PREVIO_12M_EUR'] > 0,
        100*(base['MARKET_VALUE_IN_EUR'] / base['VALOR_PREVIO_12M_EUR'] - 1),
        np.nan)
    base['DESFASE_PREVIO_12M_DIAS'] = (
        base['FECHA_VALORACION_PREVIA'] - base['FECHA_OBJETIVO_PREVIA']).dt.days
    return base


def enriquecer_jugadores(base, jugadores):
    j = leer_csv(jugadores, ['PLAYER_ID', 'DATE_OF_BIRTH'])
    j['PLAYER_ID'] = pd.to_numeric(j['PLAYER_ID'], errors='coerce')
    j['DATE_OF_BIRTH'] = pd.to_datetime(j['DATE_OF_BIRTH'], errors='coerce')
    if j['PLAYER_ID'].duplicated().any():
        raise ValueError('DIM_PLAYER contiene PLAYER_ID duplicados.')
    use = ['PLAYER_ID','DATE_OF_BIRTH']
    for col in ['FOOT','HEIGHT_IN_CM','POSITION','SUB_POSITION']:
        if col in j: use.append(col)
    base = base.merge(j[use], on='PLAYER_ID', how='left', validate='many_to_one')
    base['EDAD_ANIOS'] = (base['VALUATION_DATE'] - base['DATE_OF_BIRTH']).dt.days / 365.2425
    base.loc[~base['EDAD_ANIOS'].between(14, 60), 'EDAD_ANIOS'] = np.nan
    return base


def enriquecer_partidos(base, partidos, ventanas=(180,365)):
    p = leer_csv(partidos, ['PLAYER_ID', 'GAME_DATE'])
    p['PLAYER_ID'] = pd.to_numeric(p['PLAYER_ID'], errors='coerce')
    p['GAME_DATE'] = pd.to_datetime(p['GAME_DATE'], errors='coerce')
    p = p.dropna(subset=['PLAYER_ID','GAME_DATE'])
    if 'GAME_ID' in p and p.duplicated(['GAME_ID','PLAYER_ID']).any():
        raise ValueError('FACT_PLAYER_MATCH viola la clave (GAME_ID, PLAYER_ID).')
    medidas = ['MINUTES_PLAYED','GOALS','ASSISTS']
    for m in medidas:
        if m not in p: raise ValueError(f'FACT_PLAYER_MATCH no contiene {m}.')
        p[m] = pd.to_numeric(p[m], errors='coerce')
    p['PARTIDOS'] = 1
    # Sumas acumuladas y contadores de valores NO NULOS por jugador y día.
    # La resta de acumulados permite consultas O(log n) sin joins explosivos.
    for m in medidas:
        p[f'{m}_VALIDOS'] = p[m].notna().astype('int64')
        p[m] = p[m].fillna(0)
    sums = ['PARTIDOS', *medidas, *(f'{m}_VALIDOS' for m in medidas)]
    diario = p.groupby(['PLAYER_ID','GAME_DATE'], as_index=False)[sums].sum()
    diario[sums] = diario.groupby('PLAYER_ID', sort=False)[sums].cumsum()
    res = base
    for dias in ventanas:
        # Intervalo [fecha_valoracion-dias, fecha_valoracion): no incluye el día del corte.
        res[f'CONSULTA_FIN_{dias}D'] = res['VALUATION_DATE'] - pd.Timedelta(days=1)
        res[f'CONSULTA_INICIO_{dias}D'] = res['VALUATION_DATE'] - pd.Timedelta(days=dias+1)
        h_fin = diario.rename(columns={c:f'{c}_FIN_{dias}D' for c in sums})
        h_inicio = diario.rename(columns={c:f'{c}_INICIO_{dias}D' for c in sums})
        res = asof_por_jugador(res, h_fin, f'CONSULTA_FIN_{dias}D', 'GAME_DATE')
        res = res.drop(columns=['GAME_DATE'])
        res = asof_por_jugador(res, h_inicio, f'CONSULTA_INICIO_{dias}D', 'GAME_DATE')
        res = res.drop(columns=['GAME_DATE'])
        for m in sums:
            res[f'{m}_{dias}D'] = (res[f'{m}_FIN_{dias}D'].fillna(0) -
                                   res[f'{m}_INICIO_{dias}D'].fillna(0))
        # Si un jugador no tiene partidos observados, desconocemos su actividad,
        # por lo que las medidas quedan NA, no 0.
        observados = res[f'PARTIDOS_{dias}D'] > 0
        res[f'TIENE_PARTIDOS_{dias}D'] = observados.astype('int8')
        res.loc[~observados, [f'{m}_{dias}D' for m in medidas]] = np.nan
        # Si sí hay partidos, pero ninguna cifra disponible para la medida,
        # tampoco imputamos cero.
        for m in medidas:
            res.loc[res[f'{m}_VALIDOS_{dias}D'] == 0, f'{m}_{dias}D'] = np.nan
        res = res.drop(columns=[
            f'CONSULTA_FIN_{dias}D', f'CONSULTA_INICIO_{dias}D',
            *(f'{c}_{s}_{dias}D' for c in sums for s in ('FIN','INICIO')),
        ])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--etiquetas', type=Path, default=Path('output/valuaciones_etiquetadas_u40.csv'))
    ap.add_argument('--valoraciones', type=Path, default=Path('our_dataset/fact_player_valuation.csv'))
    ap.add_argument('--jugadores', type=Path, default=Path('our_dataset/dim_player.csv'))
    ap.add_argument('--partidos', type=Path, default=Path('our_dataset/fact_player_match.csv'))
    ap.add_argument('--salida', type=Path, default=Path('output'))
    ap.add_argument('--tolerancia-previa-dias', type=int, default=90)
    ap.add_argument('--target', default='TARGET_DEPRECIACION_40')
    args = ap.parse_args()
    if args.tolerancia_previa_dias < 0 or args.tolerancia_previa_dias >= 365:
        ap.error('--tolerancia-previa-dias debe estar entre 0 y 364')
    base = leer_csv(args.etiquetas, ['PLAYER_ID','VALUATION_DATE','MARKET_VALUE_IN_EUR',args.target])
    base['VALUATION_DATE'] = pd.to_datetime(base['VALUATION_DATE'], errors='coerce')
    base['PLAYER_ID'] = pd.to_numeric(base['PLAYER_ID'], errors='coerce')
    if base[['PLAYER_ID','VALUATION_DATE']].isna().any().any():
        raise ValueError('Hay identificadores o fechas inválidas en etiquetas.')
    if base.duplicated(['PLAYER_ID','VALUATION_DATE']).any():
        raise ValueError('La tabla etiquetada repite el grain jugador-fecha.')
    if not base[args.target].isin([0,1]).all():
        raise ValueError('Se esperaba una tabla etiquetada con target binario 0/1.')
    n = len(base)
    print('Etiquetas de entrada:', n)
    base = enriquecer_valoraciones(base, args.valoraciones, args.tolerancia_previa_dias)
    base = enriquecer_jugadores(base, args.jugadores)
    base = enriquecer_partidos(base, args.partidos)
    assert len(base) == n and not base.duplicated(['PLAYER_ID','VALUATION_DATE']).any()
    args.salida.mkdir(parents=True, exist_ok=True)
    base = base.sort_values(['VALUATION_DATE','PLAYER_ID'])
    base.to_csv(args.salida/'auditoria_enriquecida.csv', index=False)
    features = ['PLAYER_ID','VALUATION_DATE','CLUB_ID','COMPETITION_ID',
                'EDAD_ANIOS','FOOT','HEIGHT_IN_CM','POSITION','SUB_POSITION',
                'MARKET_VALUE_IN_EUR','VALOR_PREVIO_12M_EUR',
                'VARIACION_PREVIA_12M_PCT','DESFASE_PREVIO_12M_DIAS']
    features = [c for c in features if c in base]
    for d in (180,365):
        features += [f'PARTIDOS_{d}D', f'TIENE_PARTIDOS_{d}D']
        features += [f'{m}_{d}D' for m in ['MINUTES_PLAYED','GOALS','ASSISTS']]
    features += [args.target]
    # PLAYER_ID y VALUATION_DATE son columnas de identidad y partición temporal,
    # no introducirlas automáticamente a X en el entrenamiento.
    base[features].to_csv(args.salida/'modelo_etiquetado.csv', index=False)
    lineas = [f'Filas: {len(base):,}', f'Jugadores: {base.PLAYER_ID.nunique():,}',
              f'Distribución target: {base[args.target].value_counts().to_dict()}']
    for d in (180,365):
        lineas.append(f'Con al menos un partido en {d}d: {int(base[f"TIENE_PARTIDOS_{d}D"].sum()):,} ({base[f"TIENE_PARTIDOS_{d}D"].mean()*100:.2f}%)')
    lineas += [f'Sin edad: {base.EDAD_ANIOS.isna().sum():,}',
               f'Sin variación previa: {base.VARIACION_PREVIA_12M_PCT.isna().sum():,}',
               'IMPORTANTE: eliminar PLAYER_ID y VALUATION_DATE de X; usar la fecha para partición temporal.',
               'No incorporar las columnas FECHA_VALUACION_FUTURA, VALOR_FUTURO_EUR, PROPORCION_VALOR, VARIACION_PCT o N_VALUACIONES_HORIZONTE como X.',
               'La ausencia de partidos en Gold no prueba ausencia real de actividad.']
    reporte='\n'.join(lineas)+'\n'
    (args.salida/'resumen.txt').write_text(reporte, encoding='utf-8')
    print(reporte)
    print('Guardados en:',args.salida)

if __name__ == '__main__': main()
