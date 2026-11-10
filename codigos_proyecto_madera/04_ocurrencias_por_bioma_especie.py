"""Agrupa las ocurrencias por especie contando ocurrencias por bioma y etiqueta.

Parte de Data_ocurrencias/ocurrencias_con_bioma.csv (salida de
03_asignar_biomas.py) y genera una tabla con una fila por especie y dos
columnas de conteo por cada bioma de la IUCN, según la etiqueta del
ráster (1 = ocurrencia mayor, 2 = ocurrencia menor):

  CONTEOS_T1.1_mayor, CONTEOS_T1.1_menor,
  CONTEOS_T1.2_mayor, CONTEOS_T1.2_menor,
  CONTEOS_T1.3_mayor, CONTEOS_T1.3_menor

Cada ocurrencia puede sumar en más de una columna (los mapas se solapan)
o en ninguna (fuera de los tres biomas). Los porcentajes se calculan
SOLO sobre las ocurrencias que caen en al menos un bioma
(OCURRENCIAS_CON_BIOMA); las que quedan fuera de los tres mapas se
omiten del denominador y no influyen en el bioma final. Aun así los
porcentajes pueden no sumar 100 por los solapamientos.

Equivale a count_occurrences_by_biome*.py del pipeline antiguo de
Machine_wood_models/retrieve_biome, adaptado a los TIFF de Biomas_zonas.
"""

import os

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, 'Data_ocurrencias')

INPUT_PATH = os.path.join(DATA_DIR, 'ocurrencias_con_bioma.csv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'ocurrencias_por_bioma_especie_con_porcentaje.csv')

BIOMAS = {'T1_1': 'T1.1', 'T1_2': 'T1.2', 'T1_3': 'T1.3'}
ETIQUETAS = {1: 'mayor', 2: 'menor'}


def main():
    df = pd.read_csv(INPUT_PATH, usecols=['species'] + list(BIOMAS))
    print(f"Registros: {len(df)}")
    print(f"Especies únicas: {df['species'].nunique()}")

    grupos = df.groupby('species')
    resultado = pd.DataFrame(index=grupos.size().index)

    conteo_cols = []
    for codigo, nombre in BIOMAS.items():
        for valor, etiqueta in ETIQUETAS.items():
            col = f'CONTEOS_{nombre}_{etiqueta}'
            resultado[col] = grupos[codigo].apply(lambda s, v=valor: (s == v).sum())
            conteo_cols.append(col)

    resultado['TOTAL_OCURRENCIAS'] = grupos.size()

    # Ocurrencias que caen en al menos un bioma: son el denominador de los
    # porcentajes; las que quedan fuera de los tres mapas se omiten.
    df['con_bioma'] = (df[list(BIOMAS)] > 0).any(axis=1)
    resultado['OCURRENCIAS_CON_BIOMA'] = df.groupby('species')['con_bioma'].sum()

    for col in conteo_cols:
        pct_col = col.replace('CONTEOS_', 'PORCENTAJE_')
        resultado[pct_col] = (
            resultado[col] / resultado['OCURRENCIAS_CON_BIOMA'] * 100
        ).round(2).fillna(0)

    # Ordenar columnas: conteos, porcentajes y totales al final
    orden = conteo_cols + [c.replace('CONTEOS_', 'PORCENTAJE_') for c in conteo_cols]
    resultado = resultado[orden + ['OCURRENCIAS_CON_BIOMA', 'TOTAL_OCURRENCIAS']]

    resultado.to_csv(OUTPUT_PATH)
    print(f"\nGuardado: {OUTPUT_PATH}")
    print(f"Especies en la tabla: {len(resultado)}")
    print("\nEjemplo:")
    print(resultado.head(5).to_string())


if __name__ == '__main__':
    main()
