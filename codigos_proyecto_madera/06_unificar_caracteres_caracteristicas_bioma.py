"""Une el bioma final por especie con las características de InsideWood.

Toma el bioma_final de Data_ocurrencias/ocurrencias_con_bioma_final_<umbral>.csv
(salida de 05_definir_bioma_final.py) y lo agrega mediante LEFT JOIN a
caracteristicas/InsideWood_Humboldt_Limpio.xlsx (columna 'Species').

La tabla de características se mantiene como base: todas sus filas se
conservan y las especies sin bioma final quedan con NaN.

Regla especial (heredada de Machine_wood_models/retrieve_biome/
merge_biome_features_4_labels_updated.py): toda fila con Source = Humboldt
recibe bioma_final = T1.2 (bosque seco tropical) con máxima prioridad, aunque
las ocurrencias de la especie indiquen otro bioma o no existan. La columna
'bioma_origen' indica si el bioma viene de 'regla_humboldt' o de
'ocurrencias'.

Genera un archivo por umbral en Data_ocurrencias:

  caracteristicas_InsideWood_Limpio_con_bioma_<umbral>.csv

Nota: los caracteres de caracteres/WoodAnatomy.csv NO se unen aquí porque
sus 1417 especies no se solapan con las 751 de Data_ocurrencias (solo
comparten géneros). Su bioma se calcula desde sus propias ocurrencias en
07_biomas_para_woodanatomy.py.
"""

import os

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, 'Data_ocurrencias')

CARACTERISTICAS_PATH = os.path.join(
    BASE_DIR, 'caracteristicas', 'InsideWood_Humboldt_Limpio.xlsx'
)

UMBRALES = [60, 80]

# Columnas del archivo de biomas que se agregan a cada dataset
COLUMNAS_BIOMA = ['bioma_final', 'porcentaje_maximo', 'etiqueta_dominante']

# Regla Humboldt: las filas de esta fuente se reportan en bosque seco tropical
FUENTE_HUMBOLDT = 'humboldt'
BIOMA_HUMBOLDT = 'T1.2'


def normalizar_especie(serie):
    return (
        serie.astype(str)
        .str.replace('_', ' ', regex=False)
        .str.strip()
        .str.lower()
    )


def cargar_biomas(umbral):
    path = os.path.join(DATA_DIR, f'ocurrencias_con_bioma_final_{umbral}.csv')
    biomas = pd.read_csv(path)[['species'] + COLUMNAS_BIOMA]
    biomas['species_norm'] = normalizar_especie(biomas['species'])
    return biomas.drop(columns=['species'])


def unir(df_base, col_especie, biomas):
    df = df_base.copy()
    df['species_norm'] = normalizar_especie(df[col_especie])
    df = df.merge(biomas, on='species_norm', how='left')
    return df.drop(columns=['species_norm'])


def aplicar_regla_humboldt(df):
    df = df.copy()
    es_humboldt = df['Source'].astype(str).str.strip().str.lower() == FUENTE_HUMBOLDT
    previo = df['bioma_final']
    conflicto = es_humboldt & previo.notna() & (previo != BIOMA_HUMBOLDT)
    df['bioma_origen'] = previo.notna().map({True: 'ocurrencias', False: None})
    df.loc[es_humboldt, 'bioma_final'] = BIOMA_HUMBOLDT
    df.loc[es_humboldt, 'bioma_origen'] = 'regla_humboldt'
    print(f"  Regla Humboldt: {es_humboldt.sum()} filas -> {BIOMA_HUMBOLDT} "
          f"({previo[es_humboldt].isna().sum()} sin bioma previo, "
          f"{conflicto.sum()} contradecían las ocurrencias)")
    return df


def reportar(nombre, df):
    con_bioma = df['bioma_final'].notna().sum()
    print(f"  {nombre}: {len(df)} filas, {con_bioma} con bioma_final "
          f"({con_bioma / len(df) * 100:.1f}%)")
    print(f"    Distribución: {df['bioma_final'].value_counts().to_dict()}")


def main():
    caracteristicas = pd.read_excel(CARACTERISTICAS_PATH)
    print(f"Características (InsideWood Limpio): {caracteristicas.shape}")

    for umbral in UMBRALES:
        print(f"\n=== Umbral {umbral}%")
        biomas = cargar_biomas(umbral)

        df_feat = unir(caracteristicas, 'Species', biomas)
        df_feat = aplicar_regla_humboldt(df_feat)
        out_feat = os.path.join(
            DATA_DIR, f'caracteristicas_InsideWood_Limpio_con_bioma_{umbral}.csv'
        )
        df_feat.to_csv(out_feat, index=False)
        reportar(f'características -> {os.path.basename(out_feat)}', df_feat)


if __name__ == '__main__':
    main()
