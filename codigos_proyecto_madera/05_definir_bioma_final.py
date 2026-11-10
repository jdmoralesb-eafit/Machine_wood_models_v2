"""Define el bioma final de cada especie agrupando las etiquetas mayor y menor.

Parte de Data_ocurrencias/ocurrencias_por_bioma_especie_con_porcentaje.csv
(salida de 04_ocurrencias_por_bioma_especie.py). Para cada bioma se suman
los porcentajes de ocurrencia mayor y menor (cada ocurrencia tiene un solo
valor por ráster, así que no hay doble conteo dentro de un bioma):

    PORCENTAJE_T1.x_total = PORCENTAJE_T1.x_mayor + PORCENTAJE_T1.x_menor

El bioma final es el de mayor porcentaje total, siempre que supere el
umbral; si no lo supera, la especie queda sin bioma final. Se guarda
también 'etiqueta_dominante' (si dentro del bioma ganador pesa más la
zona de ocurrencia mayor o la menor) y 'porcentaje_maximo'.

Genera un archivo por cada umbral (60% y 80%).

La versión anterior, que comparaba las seis columnas mayor/menor por
separado contra el umbral, está respaldada en
05_definir_bioma_final_backup_diferenciando_mayor_menor.py.
"""

import os

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, 'Data_ocurrencias')

INPUT_PATH = os.path.join(DATA_DIR, 'ocurrencias_por_bioma_especie_con_porcentaje.csv')

BIOMAS = ['T1.1', 'T1.2', 'T1.3']
UMBRALES = [60, 80]


def definir_bioma_final(df, umbral):
    resultado = df.copy()

    total_cols = []
    for bioma in BIOMAS:
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

    # Dentro del bioma ganador, ¿pesa más la zona mayor o la menor?
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
    df = pd.read_csv(INPUT_PATH, index_col='species')
    print(f"Especies: {len(df)}")

    for umbral in UMBRALES:
        resultado = definir_bioma_final(df, umbral)
        output_path = os.path.join(
            DATA_DIR, f'ocurrencias_con_bioma_final_{umbral}.csv'
        )
        resultado.to_csv(output_path)

        print(f"\n=== Umbral {umbral}% -> {output_path}")
        print(f"Especies con bioma final: {resultado['bioma_final'].notna().sum()}")
        print(f"Especies sin bioma final: {resultado['bioma_final'].isna().sum()}")
        print("Distribución de bioma_final:")
        print(resultado['bioma_final'].value_counts(dropna=False).to_string())
        print("Etiqueta dominante dentro del bioma ganador:")
        print(resultado['etiqueta_dominante'].value_counts(dropna=False).to_string())


if __name__ == '__main__':
    main()
