"""Une las dos fuentes de ocurrencias (BIEN) en un solo CSV (versión final).

Fuentes (Machine_wood_models/retrieve_biome):
  - data_obtain_input_files/species_coordinates_v2.csv  (802 especies, abril 2025)
  - coordinates_all_species.csv                         (1738 especies, mayo 2025)

Se concatenan y se eliminan los puntos duplicados (misma especie y mismas
coordenadas redondeadas a 5 decimales ~ 1 m).
Salida: Data_ocurrencias/ocurrencias_unificadas.csv
"""
import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RB = os.path.join(BASE_DIR, 'Machine_wood_models', 'retrieve_biome')
DATA_DIR = os.path.join(BASE_DIR, 'Data_ocurrencias')
OUTPUT_PATH = os.path.join(DATA_DIR, 'ocurrencias_unificadas.csv')
COLS = ['species', 'longitude', 'latitude']


def leer(path, fuente):
    df = pd.read_csv(path, usecols=COLS).dropna()
    df['species'] = df['species'].str.strip()
    df['longitude'] = df['longitude'].round(5)
    df['latitude'] = df['latitude'].round(5)
    df = df.drop_duplicates()
    print(f"{fuente}: {len(df)} puntos únicos, {df['species'].nunique()} especies")
    return df


def main():
    v2 = leer(os.path.join(RB, 'data_obtain_input_files', 'species_coordinates_v2.csv'), 'v2')
    al = leer(os.path.join(RB, 'coordinates_all_species.csv'), 'all')
    tot = pd.concat([v2, al], ignore_index=True).drop_duplicates().sort_values(COLS).reset_index(drop=True)
    print(f"\nUnión: {len(tot)} puntos únicos, {tot['species'].nunique()} especies")
    os.makedirs(DATA_DIR, exist_ok=True)
    tot.to_csv(OUTPUT_PATH, index=False)
    print(f"Guardado: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
