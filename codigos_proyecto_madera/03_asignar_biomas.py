"""Asigna a cada ocurrencia un bioma según los TIFF de Biomas_zonas.

Los tres TIFF son mapas indicativos de la tipología global de ecosistemas
de la IUCN (valores: 0 = ausente/nodata, 1 = ocurrencia mayor,
2 = ocurrencia menor):

  - T1.1.web.mix_v2.0.tif  -> T1.1 Selva húmeda tropical de tierras bajas (EPSG:4326)
  - T1.2.web.mix_v1.0.tif  -> T1.2 Bosque seco tropical (EPSG:4326)
  - T1.3.WM.nwx_v1.0.tif   -> T1.3 Selva húmeda tropical de montaña (EPSG:4087)

Para cada punto se muestrean los tres rásteres (lectura vectorizada por
ventana) y se asigna el bioma con regla de prioridad:

  1. Si el punto cae en zona de ocurrencia mayor (valor 1) de un solo
     bioma, se asigna ese bioma.
  2. Si cae en ocurrencia mayor de varios, se marca como empate y no se
     asigna bioma único (queda registrado en la columna del detalle).
  3. Si no cae en ninguna mayor pero sí en una menor (valor 2), se asigna
     esa; con empate entre menores se aplica lo mismo que en 2.
  4. Si no cae en ninguno, bioma_asignado queda vacío.

Se conservan las columnas T1_1, T1_2 y T1_3 con el valor crudo de cada
ráster para poder aplicar otras reglas sin volver a muestrear.
"""

import os

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, 'Data_ocurrencias')
TIFF_DIR = os.path.join(BASE_DIR, 'Biomas_zonas')

INPUT_PATH = os.path.join(DATA_DIR, 'ocurrencias_unificadas.csv')

# Solo estas columnas se leen del archivo de entrada; cualquier otra
# (p. ej. region_ecologica) se descarta.
COLUMNAS_ENTRADA = ['species', 'longitude', 'latitude']
OUTPUT_PATH = os.path.join(DATA_DIR, 'ocurrencias_con_bioma.csv')

RASTERS = [
    ('T1_1', 'T1.1.web.mix_v2.0.tif'),
    ('T1_2', 'T1.2.web.mix_v1.0.tif'),
    ('T1_3', 'T1.3.WM.nwx_v1.0.tif'),
]

BIOME_LABELS = {'T1_1': 'T1.1', 'T1_2': 'T1.2', 'T1_3': 'T1.3'}


def muestrear_raster(tif_path, lons, lats):
    """Devuelve el valor del ráster para cada punto (0 donde cae fuera).

    Lee solo la ventana que cubre los puntos en lugar del ráster completo.
    """
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


def asignar_bioma(df):
    """Aplica la regla de prioridad mayor > menor sobre las columnas T1_x."""
    codigos = list(BIOME_LABELS)
    valores = df[codigos].to_numpy()

    bioma = np.full(len(df), None, dtype=object)
    detalle = np.full(len(df), 'sin_bioma', dtype=object)

    for nivel, nombre_nivel in ((1, 'mayor'), (2, 'menor')):
        en_nivel = valores == nivel
        n_biomas = en_nivel.sum(axis=1)

        unico = (bioma == None) & (n_biomas == 1)  # noqa: E711
        idx_bioma = en_nivel.argmax(axis=1)
        bioma[unico] = [BIOME_LABELS[codigos[i]] for i in idx_bioma[unico]]
        detalle[unico] = nombre_nivel

        empate = (bioma == None) & (detalle == 'sin_bioma') & (n_biomas > 1)  # noqa: E711
        detalle[empate] = f'empate_{nombre_nivel}'

    df['bioma_asignado'] = bioma
    df['asignacion_detalle'] = detalle
    return df


def main():
    df = pd.read_csv(INPUT_PATH, usecols=COLUMNAS_ENTRADA)[COLUMNAS_ENTRADA]
    print(f"Registros: {len(df)}")

    coords_ok = df['longitude'].notna() & df['latitude'].notna()
    lons = df.loc[coords_ok, 'longitude'].to_numpy(dtype=np.float64)
    lats = df.loc[coords_ok, 'latitude'].to_numpy(dtype=np.float64)

    for codigo, tif_name in RASTERS:
        print(f"Muestreando {tif_name} ...")
        df[codigo] = 0
        df.loc[coords_ok, codigo] = muestrear_raster(
            os.path.join(TIFF_DIR, tif_name), lons, lats
        )

    df = asignar_bioma(df)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nGuardado: {OUTPUT_PATH}")
    print("\nDistribución de bioma_asignado:")
    print(df['bioma_asignado'].value_counts(dropna=False))
    print("\nDetalle de la asignación:")
    print(df['asignacion_detalle'].value_counts())


if __name__ == '__main__':
    main()
