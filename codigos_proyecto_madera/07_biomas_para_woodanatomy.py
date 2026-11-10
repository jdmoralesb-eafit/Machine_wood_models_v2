"""Asigna biomas a las especies de WoodAnatomy y une el resultado con sus caracteres.

Las 1417 especies de caracteres/WoodAnatomy.csv no se solapan con las 751
de Data_ocurrencias, así que su bioma no puede venir de ahí. Este script
repite el pipeline completo (03 -> 04 -> 05) usando las ocurrencias de
Machine_wood_models/retrieve_biome/coordinates_all_species.csv, que sí
contiene coordenadas para 987 de esas especies:

  1. Filtra las ocurrencias a las especies presentes en WoodAnatomy.
  2. Muestrea los tres TIFF de Biomas_zonas (1 = ocurrencia mayor,
     2 = menor) igual que 03_asignar_biomas.py.
  3. Agrupa por especie con dos columnas de conteo por bioma y calcula
     porcentajes solo sobre las ocurrencias que caen en algún bioma,
     igual que 04_ocurrencias_por_bioma_especie.py.
  4. Define el bioma final sumando mayor+menor por bioma contra los
     umbrales de 60% y 80%, igual que 05_definir_bioma_final.py.
  5. Une el bioma final a WoodAnatomy.csv mediante LEFT JOIN (los nombres
     Genero_epiteto se normalizan a espacios y minúsculas para el cruce).

Salidas en Data_ocurrencias:

  woodanatomy_por_bioma_especie_con_porcentaje.csv
  caracteres_WoodAnatomy_con_bioma_60.csv
  caracteres_WoodAnatomy_con_bioma_80.csv
"""

import os

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, 'Data_ocurrencias')
TIFF_DIR = os.path.join(BASE_DIR, 'Biomas_zonas')

CARACTERES_PATH = os.path.join(BASE_DIR, 'caracteres', 'WoodAnatomy.csv')
COORDS_PATH = os.path.join(DATA_DIR, 'ocurrencias_unificadas.csv')
OUTPUT_PORCENTAJES = os.path.join(
    DATA_DIR, 'woodanatomy_por_bioma_especie_con_porcentaje.csv'
)

RASTERS = [
    ('T1_1', 'T1.1.web.mix_v2.0.tif'),
    ('T1_2', 'T1.2.web.mix_v1.0.tif'),
    ('T1_3', 'T1.3.WM.nwx_v1.0.tif'),
]
BIOMAS = {'T1_1': 'T1.1', 'T1_2': 'T1.2', 'T1_3': 'T1.3'}
ETIQUETAS = {1: 'mayor', 2: 'menor'}
UMBRALES = [60, 80]


def normalizar_especie(serie):
    return (
        serie.astype(str)
        .str.replace('_', ' ', regex=False)
        .str.strip()
        .str.lower()
    )


def muestrear_raster(tif_path, lons, lats):
    """Devuelve el valor del ráster para cada punto (0 donde cae fuera)."""
    valores = np.zeros(len(lons), dtype=np.int16)

    with rasterio.open(tif_path) as src:
        if src.crs.to_epsg() == 4326:
            xs, ys = lons, lats
        else:
            transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
            xs, ys = transformer.transform(lons, lats)

        cols, rows = (~src.transform) * (xs, ys)
        cols = np.floor(cols).astype(np.int64)
        rows = np.floor(rows).astype(np.int64)

        dentro = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        if not dentro.any():
            return valores

        r0, r1 = rows[dentro].min(), rows[dentro].max()
        c0, c1 = cols[dentro].min(), cols[dentro].max()
        window = rasterio.windows.Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
        bloque = src.read(1, window=window)

        valores[dentro] = bloque[rows[dentro] - r0, cols[dentro] - c0]

    return valores


def agrupar_por_especie(df):
    grupos = df.groupby('species_norm')
    resultado = pd.DataFrame(index=grupos.size().index)

    conteo_cols = []
    for codigo, nombre in BIOMAS.items():
        for valor, etiqueta in ETIQUETAS.items():
            col = f'CONTEOS_{nombre}_{etiqueta}'
            resultado[col] = grupos[codigo].apply(lambda s, v=valor: (s == v).sum())
            conteo_cols.append(col)

    resultado['TOTAL_OCURRENCIAS'] = grupos.size()

    df['con_bioma'] = (df[list(BIOMAS)] > 0).any(axis=1)
    resultado['OCURRENCIAS_CON_BIOMA'] = grupos['con_bioma'].sum()

    for col in conteo_cols:
        pct_col = col.replace('CONTEOS_', 'PORCENTAJE_')
        resultado[pct_col] = (
            resultado[col] / resultado['OCURRENCIAS_CON_BIOMA'] * 100
        ).round(2).fillna(0)

    orden = conteo_cols + [c.replace('CONTEOS_', 'PORCENTAJE_') for c in conteo_cols]
    return resultado[orden + ['OCURRENCIAS_CON_BIOMA', 'TOTAL_OCURRENCIAS']]


def definir_bioma_final(df, umbral):
    resultado = df.copy()

    total_cols = []
    for bioma in BIOMAS.values():
        col = f'PORCENTAJE_{bioma}_total'
        resultado[col] = (
            resultado[f'PORCENTAJE_{bioma}_mayor']
            + resultado[f'PORCENTAJE_{bioma}_menor']
        )
        total_cols.append(col)

    maximos = resultado[total_cols].max(axis=1)
    ganadores = (
        resultado[total_cols].idxmax(axis=1)
        .str.replace('PORCENTAJE_', '', regex=False)
        .str.replace('_total', '', regex=False)
    )

    supera = maximos > umbral
    resultado['porcentaje_maximo'] = maximos
    resultado['bioma_final'] = ganadores.where(supera)

    pct_mayor = resultado.to_numpy()[
        range(len(resultado)),
        [resultado.columns.get_loc(f'PORCENTAJE_{b}_mayor') for b in ganadores]
    ]
    pct_menor = resultado.to_numpy()[
        range(len(resultado)),
        [resultado.columns.get_loc(f'PORCENTAJE_{b}_menor') for b in ganadores]
    ]
    etiqueta = pd.Series(
        ['mayor' if m >= n else 'menor' for m, n in zip(pct_mayor, pct_menor)],
        index=resultado.index,
    )
    resultado['etiqueta_dominante'] = etiqueta.where(supera)

    return resultado


def main():
    caracteres = pd.read_csv(CARACTERES_PATH, encoding='latin-1', sep=';')
    caracteres['species_norm'] = normalizar_especie(caracteres['sp'])
    print(f"WoodAnatomy: {len(caracteres)} especies")

    coords = pd.read_csv(COORDS_PATH, usecols=['species', 'longitude', 'latitude'])
    coords['species_norm'] = normalizar_especie(coords['species'])
    coords = coords[coords['species_norm'].isin(set(caracteres['species_norm']))]
    coords = coords.dropna(subset=['longitude', 'latitude']).reset_index(drop=True)
    print(f"Ocurrencias para especies de WoodAnatomy: {len(coords)} "
          f"({coords['species_norm'].nunique()} especies con coordenadas)")

    lons = coords['longitude'].to_numpy(dtype=np.float64)
    lats = coords['latitude'].to_numpy(dtype=np.float64)
    for codigo, tif_name in RASTERS:
        print(f"Muestreando {tif_name} ...")
        coords[codigo] = muestrear_raster(
            os.path.join(TIFF_DIR, tif_name), lons, lats
        )

    por_especie = agrupar_por_especie(coords)
    por_especie.index.name = 'species'
    por_especie.to_csv(OUTPUT_PORCENTAJES)
    print(f"\nGuardado: {OUTPUT_PORCENTAJES}")

    for umbral in UMBRALES:
        finales = definir_bioma_final(por_especie, umbral)
        cols_bioma = ['bioma_final', 'porcentaje_maximo', 'etiqueta_dominante',
                      'OCURRENCIAS_CON_BIOMA', 'TOTAL_OCURRENCIAS']
        unido = caracteres.merge(
            finales[cols_bioma], left_on='species_norm', right_index=True, how='left'
        ).drop(columns=['species_norm'])

        output_path = os.path.join(
            DATA_DIR, f'caracteres_WoodAnatomy_con_bioma_{umbral}.csv'
        )
        unido.to_csv(output_path, index=False)

        con_bioma = unido['bioma_final'].notna().sum()
        print(f"\n=== Umbral {umbral}% -> {output_path}")
        print(f"Especies con bioma final: {con_bioma} de {len(unido)} "
              f"({con_bioma / len(unido) * 100:.1f}%)")
        print("Distribución de bioma_final:")
        print(unido['bioma_final'].value_counts(dropna=False).to_string())


if __name__ == '__main__':
    main()
