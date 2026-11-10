"""Carga y preparacion de datos para los modelos: lectura de las tablas por umbral, seleccion de predictoras
y binarizacion (celda vacia -> 0, no vacia -> 1) verificada contra InsideWood_Humboldt_Procesado.xlsx."""
import os
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT, "Data_ocurrencias")
PROCESADO = os.path.join(ROOT, "caracteristicas", "InsideWood_Humboldt_Procesado.xlsx")
ORIG_TROPICAL = os.path.join(ROOT, "caracteres", "tropical_with_biomes_final_v2_60.csv")
KEY_COLS = ['Source', 'Family', 'Genus', 'Specific epithet', 'Species']
REFERENCE_FEATURES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "characteristics_60", "Machine", "feature_names.txt")

CHARACTERS_FEATURES = [
    'Growth_Rings', 'Vessel_porosity', 'Vessel_arangement', 'Perforation_plates',
    'Intervessel_pits_arregement', 'Mean_tangential_diameter_of_vessel_lumina',
    'Septa_in_fibers', 'Fibre_wall_thickness', 'Parenchyma_Paratracheal',
    'Type_of_parenchyma_banding', 'Ray_width', 'Rays_cellular_composition']

# Columnas que nunca deben ser predictoras (identificadores, etiqueta y columnas del pipeline de biomas)
LEAK_COLUMNS = ['sp', 'gen', 'fam', 'Source', 'Family', 'Genus', 'Specific epithet', 'Species',
                'bioma_final', 'porcentaje_maximo', 'etiqueta_dominante', 'bioma_origen',
                'OCURRENCIAS_CON_BIOMA', 'TOTAL_OCURRENCIAS']

def load_data(kind, threshold):
    if kind == "characteristics":
        path = os.path.join(DATA_DIR, f"caracteristicas_InsideWood_Limpio_con_bioma_{threshold}.csv")
        df = pd.read_csv(path, encoding="utf-8")
    else:  # characters / characters_plus636
        path = os.path.join(DATA_DIR, f"caracteres_WoodAnatomy_con_bioma_{threshold}.csv")
        df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    return df

def binarize_and_check(df, features):
    """Regla de InsideWood_Humboldt_Procesado: celda vacia -> 0; cualquier celda no vacia -> 1 (incluido '0').
    Solo sobre las predictoras. Verifica contra Procesado.xlsx (clave + cumcount). Devuelve df y stats."""
    raw = df[features]
    binar = raw.notna().astype(int)
    proc = pd.read_excel(PROCESADO)
    proc.columns = [str(c).strip() for c in proc.columns]
    def keyed(d):
        k = d[KEY_COLS].astype(str).agg('|'.join, axis=1)
        return k + '#' + k.groupby(k).cumcount().astype(str)
    kn = keyed(df); kp = keyed(proc)
    proc_idx = proc.set_index(kp)[features]
    common = kn[kn.isin(proc_idx.index)]
    a = binar.loc[common.index].to_numpy()
    b = proc_idx.loc[common.values].fillna(0).astype(int).to_numpy()
    assert a.shape == b.shape and (a == b).all(), "La conversion no coincide con Procesado.xlsx"
    stats = dict(filas_en_procesado=len(common), filas_adicionales=len(df) - len(common),
                 filas_procesado_total=len(proc), celdas_comparadas=int(a.size))
    out = df.copy()
    out[features] = binar
    return out, stats

def select_features(df, kind):
    """Devuelve la lista de predictoras (mismo orden que los modelos originales)."""
    if kind.startswith("characters"):
        features = CHARACTERS_FEATURES
    else:
        with open(REFERENCE_FEATURES, encoding="utf-8", newline="") as f:
            features = [l.rstrip("\r") for l in f.read().split("\n") if l.strip()]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de referencia en los datos nuevos: {missing}")
    extra = [c for c in df.columns if c not in features and c not in LEAK_COLUMNS]
    print(f"Predictoras usadas: {len(features)} | columnas nuevas descartadas (no estaban en el original): {len(extra)}")
    if extra:
        print("  Descartadas:", ascii(extra))
    assert not set(features) & set(LEAK_COLUMNS), "Fuga de datos: predictora en LEAK_COLUMNS"
    return features, extra
