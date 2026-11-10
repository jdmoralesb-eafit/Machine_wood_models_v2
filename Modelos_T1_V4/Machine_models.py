"""V4: evaluacion por especie + afinado de hiperparametros + ponderacion de clases.

Cambios respecto a V3:
  * La validacion es StratifiedGroupKFold (5 pliegues) agrupando por especie: ninguna especie
    queda a la vez en entrenamiento y prueba. Todas las filas (muestras) se siguen usando para entrenar.
  * El afinado de hiperparametros esta anidado: se hace dentro de cada pliegue de entrenamiento
    (3 pliegues internos, tambien agrupados por especie), asi que la prueba nunca se usa para elegir.
  * Peso de clases como hiperparametro: ninguno / balanced / dos_clases (equilibra T1.1 y T1.2 y deja T1.3 en peso 1).
  * T1.3 se mantiene en el entrenamiento, pero la metrica principal se calcula solo sobre T1.1 y T1.2
    (T1.3 tiene muy pocas muestras para evaluarse con fiabilidad). Se reporta aparte.
  * Linea base: la configuracion de V3 (parametros por defecto, sin pesos) bajo la misma validacion por especie.

Uso: python3 Machine_models.py {characteristics|characters_plus636} {60|80}
"""
import os
import sys
import json
import argparse
import importlib.util
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, recall_score, f1_score, accuracy_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, "..")
SEED = 42
N_OUTER, N_INNER, N_ITER = 5, 3, 12

import datos_utils as v3  # carga y binarizacion de datos

CLASES_PRINCIPALES = ["T1.1", "T1.2"]


from v4_estimators import RF, SVM, GB


PESOS = [None, "balanced", "dos_clases"]
MODELOS = {
    # nombre: (clase, parametros V3 por defecto, rejilla de busqueda)
    "KNN": (KNeighborsClassifier, {}, {
        "n_neighbors": [3, 5, 7, 11, 15, 21, 31], "weights": ["uniform", "distance"], "p": [1, 2]}),
    "Random_Forest": (RF, {}, {
        "n_estimators": [200, 400], "max_depth": [None, 10, 20, 30], "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.3], "peso": PESOS}),
    "SVM": (SVM, {}, {
        "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01], "peso": PESOS}),
    "Gradient_Boosting": (GB, {}, {
        "n_estimators": [100, 200, 300], "learning_rate": [0.05, 0.1], "max_depth": [2, 3, 5],
        "subsample": [0.7, 1.0], "peso": PESOS}),
}


def pipe(est):
    return Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0)),
                     ("sc", StandardScaler()), ("clf", est)])


def bal_acc_principal(y_true, y_pred):
    """Media del recall de T1.1 y T1.2 (codigos 0 y 1). T1.3 no pesa; si se predice T1.3 en una fila T1.1/T1.2 es error."""
    return recall_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)


SCORER = make_scorer(bal_acc_principal)


# ---------- datos ----------
def build_dataset(kind, threshold):
    df = v3.load_data(kind, threshold)
    features, extra = v3.select_features(df, kind)
    info = {}
    if kind == "characteristics":
        df, info = v3.binarize_and_check(df, features)
        gcol = "Species"
    else:
        gcol = "sp"
    df = df.dropna(subset=["bioma_final"])
    df = df[df["bioma_final"].str.strip() != ""]
    if kind == "characters_plus636":
        n_new = len(df)
        orig = pd.read_csv(v3.ORIG_TROPICAL, encoding="utf-8")
        orig.columns = [c.strip() for c in orig.columns]
        extra_rows = orig[orig["gen"].isna() | (orig["gen"].astype(str).str.strip() == "")].copy()
        extra_rows = extra_rows.dropna(subset=features)
        extra_rows["bioma_final"] = "T1.2"
        info = dict(filas_caracteres_nuevos=n_new, filas_extra_636=len(extra_rows),
                    especies_distintas_extra=int(extra_rows["sp"].nunique()))
        df = pd.concat([df[features + ["bioma_final", gcol]],
                        extra_rows[features + ["bioma_final", gcol]]], ignore_index=True)
    groups = df[gcol].astype(str).str.strip().str.lower().to_numpy()
    X = df[features]
    le = LabelEncoder().fit(df["bioma_final"])
    assert list(le.classes_) == ["T1.1", "T1.2", "T1.3"], le.classes_
    y = le.transform(df["bioma_final"])
    return X, y, groups, le, features, info


# ---------- evaluacion ----------
def metricas(y, pred):
    m1 = np.isin(y, [0, 1])
    return {
        "acc_3clases": accuracy_score(y, pred),
        "acc_T11_T12": accuracy_score(y[m1], pred[m1]),
        "bal_acc_T11_T12": bal_acc_principal(y, pred),
        "f1_macro_T11_T12": f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0),
        "recall_T11": recall_score(y, pred, labels=[0], average="macro", zero_division=0),
        "recall_T12": recall_score(y, pred, labels=[1], average="macro", zero_division=0),
        "recall_T13": recall_score(y, pred, labels=[2], average="macro", zero_division=0),
    }


def run(kind, threshold):
    out = os.path.join(HERE, f"{kind}_{threshold}", "Machine")
    os.makedirs(out, exist_ok=True)
    X, y, groups, le, features, info = build_dataset(kind, threshold)
    n_sp = len(set(groups))
    print(f"[{kind} {threshold}] filas={len(y)} especies={n_sp} clases={dict(zip(le.classes_, np.bincount(y)))}", flush=True)

    outer = list(StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED).split(X, y, groups))
    inner = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    Xn = X.to_numpy()

    rows, resumen = [], {}
    for name, (cls, defaults, grid) in MODELOS.items():
        print(f"\n== {name}", flush=True)
        oof_base = np.full(len(y), -1)
        oof_tun = np.full(len(y), -1)
        mejores, por_pliegue = [], []
        for k, (tr, te) in enumerate(outer):
            base = pipe(cls(**defaults)).fit(Xn[tr], y[tr])
            oof_base[te] = base.predict(Xn[te])
            busq = RandomizedSearchCV(
                pipe(cls()), {f"clf__{p}": v for p, v in grid.items()}, n_iter=N_ITER, scoring=SCORER,
                cv=list(inner.split(Xn[tr], y[tr], groups[tr])), n_jobs=-1, random_state=SEED, refit=True)
            busq.fit(Xn[tr], y[tr])
            oof_tun[te] = busq.predict(Xn[te])
            mejores.append({p.replace("clf__", ""): (None if v is None else v) for p, v in busq.best_params_.items()})
            por_pliegue.append(bal_acc_principal(y[te], oof_tun[te]))
            print(f"  pliegue {k+1}/{N_OUTER}: base={bal_acc_principal(y[te], oof_base[te]):.3f} "
                  f"afinado={por_pliegue[-1]:.3f}  {mejores[-1]}", flush=True)
        mb, mt = metricas(y, oof_base), metricas(y, oof_tun)
        rows.append(dict(tipo=kind, umbral=threshold, modelo=name, version="V3_por_especie", filas=len(y), **mb))
        rows.append(dict(tipo=kind, umbral=threshold, modelo=name, version="V4_afinado", filas=len(y), **mt,
                         bal_acc_std_pliegues=float(np.std(por_pliegue))))
        print(f"  RESUMEN base:    {mb}\n  RESUMEN afinado: {mt}", flush=True)

        # Modelo final: misma busqueda sobre todos los datos (validacion interna agrupada por especie)
        final = RandomizedSearchCV(
            pipe(cls()), {f"clf__{p}": v for p, v in grid.items()}, n_iter=N_ITER, scoring=SCORER,
            cv=list(StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED).split(Xn, y, groups)),
            n_jobs=-1, random_state=SEED, refit=True).fit(Xn, y)
        mdir = os.path.join(out, name)
        os.makedirs(mdir, exist_ok=True)
        joblib.dump(final.best_estimator_, os.path.join(mdir, "model.joblib"))
        rep = classification_report(y, oof_tun, target_names=le.classes_, output_dict=True, zero_division=0)
        with open(os.path.join(mdir, "classification_report_cv_por_especie.txt"), "w", encoding="utf-8") as f:
            f.write(classification_report(y, oof_tun, target_names=le.classes_, zero_division=0, digits=3))
            f.write("\nPredicciones fuera de pliegue (anidadas, agrupadas por especie).\n")
        with open(os.path.join(mdir, "mejores_parametros.json"), "w", encoding="utf-8") as f:
            json.dump(dict(modelo_final={k.replace("clf__", ""): v for k, v in final.best_params_.items()},
                           por_pliegue_externo=mejores), f, indent=2, default=str)
        resumen[name] = rep

    pd.DataFrame(rows).to_csv(os.path.join(out, "cv_metricas.csv"), index=False)
    joblib.dump(le, os.path.join(out, "label_encoder.joblib"))
    with open(os.path.join(out, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(features))
    with open(os.path.join(out, "split_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"tipo: {kind}\numbral: {threshold}\nfilas_totales: {len(y)}\nespecies_distintas: {n_sp}\n")
        f.write(f"validacion: StratifiedGroupKFold {N_OUTER} pliegues externos, {N_INNER} internos, grupo=especie, semilla {SEED}\n")
        f.write(f"busqueda: RandomizedSearchCV n_iter={N_ITER}, scoring=exactitud balanceada T1.1/T1.2\n")
        f.write(f"clases_total: {dict(zip(le.classes_, map(int, np.bincount(y))))}\n")
        for k_, v_ in info.items():
            f.write(f"{k_}: {v_}\n")
        f.write(f"n_predictoras: {len(features)}\n")
        f.write("NOTA: metrica principal = media del recall de T1.1 y T1.2 (exactitud balanceada). T1.3 permanece en el "
                "entrenamiento pero no se usa para decidir el modelo.\n")
    print(f"\nTerminado: {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["characteristics", "characters_plus636"])
    p.add_argument("threshold", type=int, choices=[60, 80])
    a = p.parse_args()
    run(a.kind, a.threshold)
