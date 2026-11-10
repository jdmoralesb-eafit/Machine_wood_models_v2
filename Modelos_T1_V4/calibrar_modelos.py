"""Crea la version calibrada (con probabilidades confiables) de SVM, Random Forest y Gradient Boosting
para las 4 configuraciones (tipo de datos x umbral). KNN no se calibra: sus probabilidades son fracciones de vecinos.

El SVM se entreno sin probabilidades; Random Forest y Gradient Boosting las entregan pero sin calibrar. Aqui cada
modelo se reentrena con los mismos hiperparametros (mejores_parametros.json) y se calibra con escalado sigmoide
(Platt) usando validacion cruzada AGRUPADA POR ESPECIE, para que la probabilidad refleje la certeza ante especies
no vistas. Tambien se evalua el modelo calibrado con validacion anidada.

Salida: <tipo>_<umbral>/Machine/<modelo>/model_calibrado.joblib y calibracion_info.json
Uso: python3 calibrar_modelos.py
"""
import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Machine_models as M  # noqa: E402

TIPOS = ["characteristics", "characters_plus636"]
MODELOS_A_CALIBRAR = ["SVM", "Random_Forest", "Gradient_Boosting"]


def construir(modelo, params):
    cls = M.MODELOS[modelo][0]
    return M.pipe(cls(**params))


def calibrado(base, cv):
    return CalibratedClassifierCV(base, method="sigmoid", cv=cv, ensemble=False)


def main():
    for tipo, umbral, modelo in [(t, u, m) for t in TIPOS for u in (60, 80) for m in MODELOS_A_CALIBRAR]:
        out = os.path.join(HERE, f"{tipo}_{umbral}", "Machine")
        with open(os.path.join(out, modelo, "mejores_parametros.json"), encoding="utf-8") as f:
            params = json.load(f)["modelo_final"]
        X, y, groups, le, features, _ = M.build_dataset(tipo, umbral)
        Xn = X.to_numpy()
        print(f"\n[{tipo} {umbral}] {modelo} {params}", flush=True)

        # Evaluacion anidada: la calibracion se ajusta solo con los datos de entrenamiento de cada pliegue
        outer = StratifiedGroupKFold(n_splits=M.N_OUTER, shuffle=True, random_state=M.SEED)
        proba = np.zeros((len(y), len(le.classes_)))
        pred_orig = np.full(len(y), -1)
        for tr, te in outer.split(Xn, y, groups):
            cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=M.SEED).split(Xn[tr], y[tr], groups[tr]))
            cal = calibrado(construir(modelo, params), cv).fit(Xn[tr], y[tr])
            proba[te] = cal.predict_proba(Xn[te])
            pred_orig[te] = construir(modelo, params).fit(Xn[tr], y[tr]).predict(Xn[te])
        pred = proba.argmax(1)
        m_cal, m_orig = M.metricas(y, pred), M.metricas(y, pred_orig)
        conf = proba.max(1)
        acierto = pred == y
        bins = [(0, .5), (.5, .7), (.7, .9), (.9, 1.01)]
        tabla = []
        for lo, hi in bins:
            s = (conf >= lo) & (conf < hi)
            tabla.append(dict(confianza=f"{lo:.0%}-{min(hi,1):.0%}", n=int(s.sum()),
                              exactitud=float(acierto[s].mean()) if s.any() else None))
        print("  sin calibrar:", {k: round(v, 3) for k, v in m_orig.items()})
        print("  calibrado:   ", {k: round(v, 3) for k, v in m_cal.items()})
        print("  confianza vs exactitud real:", tabla, flush=True)

        # Modelo final con todos los datos
        cv = list(StratifiedGroupKFold(n_splits=M.N_OUTER, shuffle=True, random_state=M.SEED).split(Xn, y, groups))
        final = calibrado(construir(modelo, params), cv).fit(Xn, y)
        joblib.dump(final, os.path.join(out, modelo, "model_calibrado.joblib"))
        with open(os.path.join(out, modelo, "calibracion_info.json"), "w", encoding="utf-8") as f:
            json.dump(dict(tipo=tipo, umbral=umbral, modelo=modelo, metodo="sigmoid (Platt), CV agrupada por especie",
                           metricas_anidadas_sin_calibrar=m_orig, metricas_anidadas_calibrado=m_cal,
                           confianza_vs_exactitud=tabla), f, indent=2)
        print(f"  guardado en {os.path.join(out, modelo)}", flush=True)


if __name__ == "__main__":
    main()
