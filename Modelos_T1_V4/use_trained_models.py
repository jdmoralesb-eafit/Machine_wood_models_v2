"""Interfaz sencilla para usar los modelos de V4.

Se elige el modelo, el tipo de datos (caracteristicas / caracteres) y el umbral (60 / 80 %); por defecto aparece el
mejor segun cv_metricas.csv (exactitud balanceada T1.1/T1.2 con validacion por especie). Se selecciona el archivo
de entrada (.csv o .xlsx) y se guardan las predicciones. El archivo de salida conserva todas las columnas de
entrada y anade Predicted_Biome, Certeza_prediccion (probabilidad de la clase elegida) y Probabilidades (las tres
clases). SVM, Random Forest y Gradient Boosting se usan en su version calibrada (crearla con calibrar_modelos.py);
KNN entrega la fraccion de vecinos de cada clase. CSV con separador ';' en UTF-8, o .xlsx.

Uso: python3 use_trained_models.py
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # los modelos guardados referencian v4_estimators
import v4_estimators  # noqa: F401,E402

MODELOS = ["SVM", "Random_Forest", "Gradient_Boosting", "KNN"]
CALIBRADOS = {"SVM", "Random_Forest", "Gradient_Boosting"}
TIPOS = {"Caracteristicas (InsideWood)": "characteristics", "Caracteres (WoodAnatomy + 636)": "characters_plus636"}
UMBRALES = [60, 80]


def mejor_modelo(tipo):
    """(umbral, modelo) con mayor exactitud balanceada T1.1/T1.2 (V4_afinado) para ese tipo de datos."""
    d = pd.concat(pd.read_csv(os.path.join(HERE, f"{tipo}_{u}", "Machine", "cv_metricas.csv")) for u in UMBRALES)
    d = d[(d.version == "V4_afinado") & (d.tipo == tipo)].sort_values("bal_acc_T11_T12", ascending=False)
    return int(d.iloc[0]["umbral"]), d.iloc[0]["modelo"]


def _imputador(pipe):
    """El primer paso del pipeline (en el SVM calibrado esta dentro del calibrador)."""
    if hasattr(pipe, "calibrated_classifiers_"):
        pipe = pipe.calibrated_classifiers_[0].estimator
    return pipe.steps[0][1]

# Etiquetas del entrenamiento (V3/V4) -> bioma
BIOMAS = {"T1.1": "Tropical humid forest", "T1.2": "Tropical dry forest", "T1.3": "Tropical montane"}


def _pct(x):
    return f"{100 * x:.1f}".replace(".", ",") + " %"


def leer_entrada(path, features):
    """Lee .csv/.xlsx. En CSV prueba ';' y ',' (y UTF-8 / latin-1) y se queda con el que reconoce mas columnas."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        return pd.read_excel(path)
    if ext != ".csv":
        raise ValueError(f"Formato no soportado ({ext}). Use .csv o .xlsx")
    mejor, mejor_n = None, -1
    for enc in ("utf-8", "latin-1"):
        for sep in (";", ","):
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc)
            except Exception:
                continue
            n = sum(c.strip() in set(features) for c in df.columns)
            if n > mejor_n:
                mejor, mejor_n = df, n
        if mejor_n > 0:
            break
    if mejor is None:
        raise ValueError("No se pudo leer el archivo CSV.")
    return mejor


def cargar_modelo(modelo, tipo, umbral):
    base = os.path.join(HERE, f"{tipo}_{umbral}", "Machine")
    ruta = os.path.join(base, modelo, "model_calibrado.joblib" if modelo in CALIBRADOS else "model.joblib")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe {ruta}" + (". Ejecute primero calibrar_modelos.py" if modelo in CALIBRADOS else ""))
    pipe = joblib.load(ruta)
    le = joblib.load(os.path.join(base, "label_encoder.joblib"))
    with open(os.path.join(base, "feature_names.txt"), encoding="utf-8") as f:
        features = [l.rstrip("\r") for l in f.read().split("\n") if l.strip()]
    return pipe, le, features


def predecir(path_entrada, modelo, tipo, umbral):
    """Devuelve (dataframe_salida, resumen_texto)."""
    pipe, le, features = cargar_modelo(modelo, tipo, umbral)
    original = leer_entrada(path_entrada, features)
    datos = original.copy()
    datos.columns = [str(c).strip() for c in datos.columns]
    faltan = [c for c in features if c not in datos.columns]
    if len(faltan) == len(features):
        raise ValueError("Ninguna columna del archivo coincide con las que usa el modelo. "
                         "Revise que haya elegido el tipo correcto (caracteristicas o caracteres).")
    for c in faltan:  # igual que el script original: las columnas ausentes se toman como 0
        datos[c] = 0
    X = datos[features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()

    imp = _imputador(pipe)
    X = X.astype(getattr(imp, "_fit_dtype", X.dtype))  # el imputador exige el mismo dtype del ajuste
    proba = pipe.predict_proba(X)
    clases = list(le.classes_)
    codigos = [clases[j] for j in proba.argmax(1)]
    salida = original.copy()
    salida["Predicted_Biome"] = [f"{BIOMAS.get(c, c)} ({c})" for c in codigos]
    salida["Certeza_prediccion"] = [_pct(p) for p in proba.max(1)]
    salida["Probabilidades"] = [" | ".join(f"{BIOMAS.get(c, c)}: {_pct(p)}" for c, p in zip(clases, fila))
                                for fila in proba]

    lineas = [f"Modelo: {modelo} | {tipo} | umbral {umbral} %",
              f"Filas procesadas: {len(salida)}",
              f"Columnas del modelo: {len(features)} (ausentes en el archivo y puestas a 0: {len(faltan)})"]
    if faltan:
        lineas.append("  Primeras ausentes: " + "; ".join(faltan[:3]))
    lineas.append("Resumen de predicciones:")
    for k, v in salida["Predicted_Biome"].value_counts().items():
        lineas.append(f"  {k}: {v}")
    lineas.append(f"Filas con certeza < 70 %: {int((proba.max(1) < 0.7).sum())} (revisar con cuidado)")
    return salida, "\n".join(lineas)


def guardar(salida, path_salida):
    ext = os.path.splitext(path_salida)[1].lower()
    if ext == ".csv":
        salida.to_csv(path_salida, sep=";", index=False, encoding="utf-8")
    elif ext in (".xls", ".xlsx"):
        salida.to_excel(path_salida, index=False, engine="openpyxl")
    else:
        raise ValueError(f"Formato de salida no soportado ({ext}). Use .csv o .xlsx")


# ---------------- interfaz ----------------
def main():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("Clasificacion de biomas - modelos V4")
    root.geometry("640x520")
    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    v_tipo = tk.StringVar(value=list(TIPOS)[0])
    v_modelo = tk.StringVar(value=MODELOS[0])
    v_umbral = tk.IntVar(value=80)
    v_archivo = tk.StringVar(value="")

    ttk.Label(frm, text="Tipo de datos").grid(row=0, column=0, sticky="w", pady=4)
    ttk.Combobox(frm, textvariable=v_tipo, values=list(TIPOS), state="readonly", width=32).grid(row=0, column=1, sticky="w")

    ttk.Label(frm, text="Modelo").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Combobox(frm, textvariable=v_modelo, values=MODELOS, state="readonly", width=24).grid(row=1, column=1, sticky="w")

    ttk.Label(frm, text="Umbral de bioma").grid(row=2, column=0, sticky="w", pady=4)
    fu = ttk.Frame(frm)
    fu.grid(row=2, column=1, sticky="w")
    for u in UMBRALES:
        ttk.Radiobutton(fu, text=f"{u} %", value=u, variable=v_umbral).pack(side="left", padx=(0, 12))

    nota = ttk.Label(frm, foreground="#555", wraplength=600, justify="left")
    nota.grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 6))

    def mejor(*_):
        u, m = mejor_modelo(TIPOS[v_tipo.get()])
        v_modelo.set(m)
        v_umbral.set(u)

    def actualiza_nota(*_):
        u, m = mejor_modelo(TIPOS[v_tipo.get()])
        nota.config(text=f"Mejor modelo para este tipo de datos: {m}, umbral {u} % (se selecciona al cambiar el tipo). "
                         "Todos los modelos entregan la certeza de cada prediccion (calibrada salvo en KNN).")
    v_tipo.trace_add("write", lambda *_: (mejor(), actualiza_nota()))
    mejor()
    actualiza_nota()

    def elegir():
        p = filedialog.askopenfilename(title="Seleccione el archivo de entrada",
                                       filetypes=[("CSV o Excel", "*.csv *.xlsx *.xls"), ("Todos", "*.*")])
        if p:
            v_archivo.set(p)

    ttk.Button(frm, text="Seleccionar archivo de entrada…", command=elegir).grid(row=4, column=0, columnspan=2, sticky="w")
    ttk.Label(frm, textvariable=v_archivo, wraplength=600, foreground="#333").grid(row=5, column=0, columnspan=3, sticky="w", pady=4)

    log = tk.Text(frm, height=14, wrap="word", state="disabled")
    log.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
    frm.rowconfigure(8, weight=1)
    frm.columnconfigure(2, weight=1)

    def escribe(txt):
        log.config(state="normal")
        log.delete("1.0", "end")
        log.insert("end", txt)
        log.config(state="disabled")

    def ejecutar():
        if not v_archivo.get():
            messagebox.showwarning("Falta el archivo", "Primero seleccione el archivo de entrada.")
            return
        try:
            salida, resumen = predecir(v_archivo.get(), v_modelo.get(), TIPOS[v_tipo.get()], v_umbral.get())
        except Exception as e:
            escribe(f"Error: {e}")
            messagebox.showerror("Error", str(e))
            return
        base = os.path.splitext(os.path.basename(v_archivo.get()))[0]
        destino = filedialog.asksaveasfilename(
            title="Guardar resultados como", initialfile=f"predictions_{base}.csv", defaultextension=".csv",
            filetypes=[("CSV (separador ;)", "*.csv"), ("Excel", "*.xlsx")])
        if not destino:
            escribe(resumen + "\n\n(No se guardo: se cancelo el cuadro de guardado.)")
            return
        try:
            guardar(salida, destino)
        except Exception as e:
            escribe(f"Error al guardar: {e}")
            messagebox.showerror("Error al guardar", str(e))
            return
        escribe(resumen + f"\n\nResultados guardados en:\n{destino}")

    ttk.Button(frm, text="Predecir y guardar…", command=ejecutar).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
    root.mainloop()


if __name__ == "__main__":
    main()
