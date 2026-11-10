# Proyecto maderas: clasificación de biomas (T1.1, T1.2, T1.3)

Predicción del bioma de una madera a partir de sus caracteres anatómicos (InsideWood / WoodAnatomy),
usando las ocurrencias geográficas de cada especie para asignar el bioma.

## Estructura
| Carpeta | Contenido |
|---|---|
| `codigos_proyecto_madera/` | Pipeline de datos (pasos 01 a 07): ocurrencias → biomas → bioma final por especie → tablas de entrenamiento |
| `Data_ocurrencias/` | `ocurrencias_unificadas.csv` (todas las ocurrencias encontradas, ~446 mil puntos) y las tablas derivadas (umbral 60 y 80) |
| `Modelos_T1_V4/` | Modelos entrenados (KNN, Random Forest, SVM, Gradient Boosting), métricas (`cv_metricas.csv`), logs, `datos_utils.py` (carga y binarización) y `use_trained_models.py` |
| `Biomas_zonas/`, `caracteres/`, `caracteristicas/` | Insumos (mapas de biomas y tablas de caracteres) |

## Orden de ejecución
1. `02_unificar_ocurrencias.py` (necesita `Machine_wood_models/retrieve_biome`, que no está en este repo; su resultado ya está en `Data_ocurrencias/ocurrencias_unificadas.csv`)
2. `03_asignar_biomas.py`, `04_...`, `05_...`, `06_...`, `07_...`
3. `python Modelos_T1_V4/Machine_models.py {characteristics|characters_plus636} {60|80}`
4. `python Modelos_T1_V4/use_trained_models.py` para predecir con cualquier modelo (por defecto el mejor); todos entregan la certeza de la predicción. Las probabilidades de SVM, Random Forest y Gradient Boosting están calibradas con `calibrar_modelos.py` (el resultado ya está incluido).

## Modelos V4
Validación `StratifiedGroupKFold` (5 pliegues) agrupada por especie, con búsqueda de hiperparámetros anidada y peso de clases.
La métrica principal es la exactitud balanceada de T1.1 y T1.2; T1.3 se entrena pero se reporta aparte (muy pocas muestras).
Mejor configuración: `characteristics_80` (exactitud balanceada ≈ 0.93).

## Instalación
`pip install -r requirements.txt`
