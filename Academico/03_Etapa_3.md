# Qué se hará específicamente en la Etapa 3: Modelado

**Objetivo:** fijar el número a superar con dos baselines y entrenar las cuatro arquitecturas de red bajo exactamente el mismo protocolo, dejando cada experimento registrado en `reports/results.csv` con sus métricas en dev.

**La etapa está "lista" cuando:** `src/evaluate.py` (primera versión), `src/models.py` y `src/train.py` existen; `notebooks/03_baselines.ipynb` y `notebooks/04_neural_networks.ipynb` corren de principio a fin; `reports/results.csv` tiene todos los experimentos (baselines, ablaciones heredadas, arquitecturas × estrategia de costo × 3 semillas); y el conjunto de test **no se ha tocado**.

**Cómo se trabaja:** acción por acción, empezando por la 0 y por los baselines. Para cada código: en qué archivo y en qué parte del notebook va, qué se logra y por qué (según el EDA y la Etapa 2). Quien lo corre pasa las salidas y se ajusta.

**Archivos que conviene adjuntar al empezar el chat:** este documento, `README.md`, `configs/default.yaml` y `models/preprocessor_meta.json` (nombres de las 383 columnas y tamaños de vocabulario).

---

## Lo más importante que deja la Etapa 2

1. **El protocolo está fijado y verificado.** Split temporal 70 / 15 / 15 por valor de `TransactionDT`; todo el preprocesamiento se ajusta solo con train; `tests/test_no_leakage.py` pasa (13 pruebas, 3.95 s con `python -m pytest tests/test_no_leakage.py -v`). Si se cambia cualquier cosa del preprocesamiento, se vuelven a correr.
2. **Los datos ya están listos y se cargan en una línea** (ver "Estado de partida"). No hay que reajustar ni volver a procesar nada en esta etapa.
3. **El desbalance se trata en el entrenamiento, no en los datos.** Ratio en train 27.4 : 1; pesos de clase `{0: 0.518, 1: 14.217}`; `scale_pos_weight` = 27.43 para XGBoost. Dev y test conservan la distribución real.
4. **El número mínimo a superar es el piso de PR-AUC**: 0.0343 en dev (0.0352 en train, 0.0348 en test). Un modelo cuyo PR-AUC no se aleje claramente de ahí no aporta nada.
5. **Ya hay señal fuerte en los datos.** Los indicadores de nulo (en `D7` el fraude es 2.74% si falta y 21.20% si vale 0; la ausencia de `addr1`/`addr2` sube el fraude a 11.25% frente a 2.51%), los importes fuera de precios recurrentes (19.3% de las filas con 6.96% de fraude, el doble de la base), la hora relativa y la identidad. Los baselines van a capturar buena parte de esto: **la red tiene que demostrar que aporta algo sobre XGBoost**, y la comparación debe ser honesta.
6. **Hay deriva, y dev no es un espejo perfecto de test.** La tasa diaria de fraude oscila (~2-3% al inicio, ~4-5% después, con picos); en test, las categorías desconocidas de `id_31` suben de 0.19% (train) a 9.33% y `id_30` de 0.11% a 2.32%. Diferencias muy pequeñas entre modelos en dev (del orden de milésimas de PR-AUC) no son concluyentes: dev tiene solo 3,042 fraudes. Por eso se usan 3 semillas y, si hace falta, bootstrap sobre dev.
7. **Dos entradas distintas según el modelo.** Los modelos tabulares (regresión logística, XGBoost, MLPs) usan `X` (383 columnas); la MLP con embeddings usa además `idx` (13 índices enteros, 0 = desconocido).
8. **Lo que se dejó pendiente a propósito:** `requirements.txt` (al final); la variante "NaN nativos" para XGBoost (el pipeline actual siempre imputa); las `drop_candidates` no se descartaron; `corr_threshold = 0.95` no se comparó con otros valores; y el test solo se miró de forma estructural (cobertura de nulos, % desconocido, % recortado) para documentar la deriva, sin ajustar ningún parámetro con ello. **En la Etapa 3 el test no se evalúa.**
9. **Entorno de trabajo:** Windows, Python 3.12.2, entorno virtual `.DL`, PowerShell. Verificar antes de empezar que TensorFlow/Keras esté instalado para esta versión de Python (hace falta una versión reciente; en Windows nativo corre solo en CPU, así que conviene acotar el presupuesto de búsqueda).

---

## Estado de partida (lo que ya existe)

- `configs/default.yaml` con semillas, split, columnas, preprocesamiento, desbalance y costos (ver tabla de decisiones cerradas).
- `src/data.py` (`load_config()`, carga y unión de tablas), `src/split.py` (`temporal_split`, `check_split`, `split_summary`), `src/preprocessing.py` (`Preprocessor` con `fit_transform`, `transform`, `save`, `load`; `write_splits`; `load_split`) y `src/train.py` con una sola función: `compute_class_weights()`.
- `tests/test_no_leakage.py` + `pytest.ini`: 13 pruebas en verde.
- `data/processed/{train,dev,test}_{X,idx,y,meta}.parquet` (~65 MB en total) y `models/preprocessor.joblib` + `models/preprocessor_meta.json` (`feature_names`, `idx_columns`, `vocab_sizes`, `null_groups`, `log1p_columns`, `clip_z`, `dropped_by_redundancy`). Ambas carpetas van ignoradas en git.
- `reports/figures/` con las figuras `01` a `19`; `docs/model_card.md` solo con notas.

Carga en una línea (el split de test existe pero **no se carga** en esta etapa):

```python
from src.data import load_config
from src.preprocessing import load_split

cfg = load_config()
tr, dv = load_split(cfg, "train"), load_split(cfg, "dev")
X_tr, idx_tr, y_tr, meta_tr = tr["X"], tr["idx"], tr["y"]["isFraud"], tr["meta"]
X_dv, idx_dv, y_dv, meta_dv = dv["X"], dv["idx"], dv["y"]["isFraud"], dv["meta"]
```

| Split | Filas | Fraudes | Tasa | Piso PR-AUC | Días relativos |
|---|---|---|---|---|---|
| train | 413,378 | 14,538 | 3.517% | 0.0352 | 1.0 - 120.8 |
| dev | 88,581 | 3,042 | 3.434% | 0.0343 | 120.8 - 152.2 |
| test (no se usa) | 88,581 | 3,083 | 3.480% | 0.0348 | 152.2 - 183.0 |

**Matrices:** `X` tiene 383 columnas (numéricas estandarizadas o binarias, indicadores de nulo, dummies de baja cardinalidad y frecuencias de las categóricas de alta cardinalidad, tras eliminar 178 redundantes de 561). `idx` tiene 13 columnas: las 12 categóricas de alta cardinalidad (`card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, `DeviceInfo`, `id_30`, `id_31`, `id_33`, `P_emaildomain`, `R_emaildomain`) más `amt_price_point_idx`. `meta` trae `TransactionID`, `TransactionDT` y `TransactionAmt` crudo (no son variables).

**Vocabularios para los `Embedding`** (ya incluyen el índice 0 = desconocido; la lista completa está en `vocab_sizes` del JSON): `card1_idx` 946, `card2_idx` 445, `amt_price_point_idx` 295, `DeviceInfo_idx` 102, `addr1_idx` 72, `id_31_idx` 65, `id_30_idx` 58, `P_emaildomain_idx` 54, `card5_idx` 45, `id_33_idx` 45.

---

## Decisiones ya cerradas en la Etapa 2 (no se reabren sin una razón)

| Parámetro | Valor | Dónde se decidió |
|---|---|---|
| `seeds` | `[19, 69, 79]` (búsqueda con la primera; configuración final con las tres) | Acción 0 |
| `split` | 0.70 / 0.15 / 0.15, temporal | Acción 1 |
| `cost` | `c_FN = 10`, `c_FP = 1`, `target_precision = 0.5` | Acción 0 |
| `imbalance` | `strategy: class_weight`, `class_weight_mode: balanced`, `focal_gamma: 2.0`, `focal_alpha: null`, `resample: false` | Acción 9 |
| `onehot_max_cardinality` / `min_category_freq` | 10 / 50 | Acción 6 |
| `amt_price_point_min_freq` | 100 (294 precios recurrentes, 80.7% de las filas) | Acción 6 |
| `skew_threshold` / `clip_z` | 2.0 / 10 | Acción 7 |
| `corr_threshold` / `corr_protect` | 0.95 / `[has_identity]` | Acción 8 |
| `min_indicator_rows` | 50 (65 indicadores de nulo) | Acción 4 |
| `numeric_impute` | `median` (con indicador de nulo obligatorio) | Acción 5 |
| `no_scale` | `[hour_sin, hour_cos]` | Acción 7 |

---

## Hallazgos que gobiernan esta etapa

| Código | Hallazgo | Implicación |
|---|---|---|
| B1 | 3.5% de fraude (27.4 : 1 en train); piso de PR-AUC ≈ 0.034-0.035 | PR-AUC es la métrica principal; pesos de clase o focal loss; `scale_pos_weight` en XGBoost; nada de sobremuestreo |
| D1 | Tasa de fraude no estacionaria; train mezcla dos regímenes (días 1-25 en ~2-3% y después ~4-5%) | Early stopping y selección de modelo en dev; no sobreinterpretar diferencias pequeñas; 3 semillas |
| G1 | |Spearman| máx ≈ 0.40; varias `V` con Spearman ≈ 0.3 y Pearson ≈ 0.02 | La regresión logística es el piso lineal; se espera que árboles y redes la superen |
| A10 / H2 | El nulo no equivale a 0 (`D7`: 2.74% vs 21.20%); `addr1`/`addr2` ausentes: 11.25% vs 2.51% | Los indicadores ya están en `X`; la ablación de imputación mide cuánto aportan |
| C3 | Precios recurrentes con tasas muy distintas (57.95 → 0.36%, 150.00 → 7.04% en train); importes fuera de precios recurrentes: 19.3% de las filas, 6.96% de fraude | Ablación del importe (variantes a-d) |
| F2 / A5 | La mayor asociación está en variables de alta cardinalidad (`card1` V de Cramér 0.32, `DeviceInfo` 0.27, `card2` 0.23) | Justifica la MLP con embeddings frente a la MLP sobre `X` |
| Etapa 2 | `id_31` desconocido: 0.19% (train) → 0.25% (dev) → 9.33% (test); `card1` desconocido ~19% y estable | El índice 0 de `id_31` se entrena con muy pocas filas; idea opcional: familia de navegador sin versión (cambia el pipeline: solo si sobra tiempo) |
| F3 | V de Cramér < 0.03 en `card4`, `M5`, `id_23`, `id_27`, `id_34` | Candidatas a descartar: se confirman con desempeño en dev, no por Cramér |
| G2 | Redundancia entre `V` | Ya resuelto con el filtro (0.95); comparar otros umbrales es opcional |

---

## Reglas que no se rompen

1. **El test no se toca.** Se evalúa **una sola vez**, con el umbral ya elegido, en la Etapa 4. En esta etapa no se carga `test_*.parquet`.
2. Toda selección (hiperparámetros, número de épocas o árboles, variantes, umbral) se hace con **dev**.
3. Los pesos de clase se calculan con train (`compute_class_weights`); no se remuestrea ningún conjunto.
4. **Comparación justa:** todos los modelos usan las mismas filas y las mismas columnas de `X`, salvo en una ablación declarada. La red con embeddings es la única que ve `idx`.
5. **No se reajusta el preprocesador.** Se carga lo ya procesado. Una ablación que cambia el pipeline (por ejemplo, imputar con 0) se hace con un `Preprocessor` nuevo ajustado con train y guardado en una carpeta aparte, nunca sobre `data/processed/`.
6. Nada va escrito a mano en los notebooks: hiperparámetros, espacios de búsqueda y estrategias salen de `configs/default.yaml`.
7. **Todo experimento deja una fila en `reports/results.csv`**, también los malos o los que se descartan.
8. Semillas: búsqueda con `seeds[0]`; configuración final con las tres; se reporta media ± desviación.
9. Si cambia cualquier cosa de `src/preprocessing.py` o de la config del preprocesamiento, se vuelven a correr los tests anti-fuga.

---

## Orden de la etapa

| # | Acción | Dónde | ¿Entrena? |
|---|---|---|---|
| 0 | Configuración de modelado | `configs/default.yaml` | No |
| 1 | Métricas (primera versión) | `src/evaluate.py` | No |
| 2 | Registro de experimentos | `src/evaluate.py` + `reports/results.csv` | No |
| 3 | Baselines: regresión logística y XGBoost | `notebooks/03_baselines.ipynb` | Sí |
| 4 | Ablaciones del importe y la imputación (heredadas) | `notebooks/03_baselines.ipynb` | Sí |
| 5 | Ablación de `drop_candidates` | `notebooks/03_baselines.ipynb` | Sí |
| 6 | Las cuatro arquitecturas | `src/models.py` | No |
| 7 | Entrenamiento | `src/train.py` | No |
| 8 | Experimentos de redes: búsqueda y 3 semillas | `notebooks/04_neural_networks.ipynb` | Sí |
| 9 | Consolidar resultados y figuras | `notebooks/04_neural_networks.ipynb` + `reports/` | No |
| 10 | Autoencoder (solo si hay tiempo) | `src/models.py`, `src/train.py` | Sí |
| 11 | Verificación y cierre | tests + README | No |

---

## Acciones

### Acción 0. Configuración de modelado

| Qué hacer | Dónde | Qué se logra | Por qué |
|---|---|---|---|
| Añadir las secciones de evaluación, baselines y redes (hiperparámetros por defecto, espacio de búsqueda, presupuesto) | `configs/default.yaml` | Ningún hiperparámetro vive en una celda; cambiar el presupuesto de búsqueda no exige tocar código | Regla 6: nada va a mano en los notebooks |

Esqueleto propuesto (los valores marcados con `?` se deciden al implementar):

```yaml
evaluation:
  primary_metric: pr_auc
  threshold_selection: dev            # umbrales solo con dev
  bootstrap: {enabled: false, n: 200} # opcional: intervalo de PR-AUC en dev
results:
  path: reports/results.csv
  predictions_dir: reports/preds      # predicciones en dev de las configuraciones finales (ignorado en git)
baselines:
  logreg: {C_grid: [0.01, 0.1, 1, 10], max_iter: 1000}     # C se elige en dev
  xgboost:
    params: {n_estimators: 2000, learning_rate: 0.05, max_depth: 6, subsample: 0.8,
             colsample_bytree: 0.8, tree_method: hist, eval_metric: aucpr}   # ?
    early_stopping_rounds: 100
nn:
  common: {batch_size: 2048, max_epochs: 100, optimizer: adam, early_stopping_patience: 10}   # ?
  embedding_dim: "min(32, round(1.6 * vocab ** 0.56))"       # ? regla propuesta por columna
  search: {n_trials: ?, seed: 19}
  strategies: [class_weight, focal]
```

### Acción 1. Métricas (primera versión)

| Qué hacer | Dónde | Qué se logra | Por qué |
|---|---|---|---|
| Funciones: `pr_auc`, `roc_auc`, `confusion`, `precision_recall_f1(y, p, thr)`, `recall_at_precision(y, p, target)`, `best_f1_threshold`, `expected_cost(y, p, thr, c_FN, c_FP)` y `evaluate_scores(y, p, cfg)`, que devuelve un diccionario con todo | `src/evaluate.py` (archivo nuevo) | Todos los modelos se miden con el mismo código, y las métricas se pueden verificar con pruebas pequeñas | **B1:** con 27.4 : 1, la exactitud no sirve y PR-AUC es la métrica principal. El README ya compromete precision, recall, F1, ROC-AUC, matriz de confusión, recall a precisión fija y costo esperado |

Definiciones que se fijan aquí:

- `pr_auc` = `average_precision_score`. `recall_at_precision` = el mayor recall entre los umbrales con precisión ≥ `target_precision` (0.5); si ninguno la alcanza, 0.
- En esta etapa los umbrales (para F1, para recall a precisión fija y para el mínimo costo) **se eligen en dev y se reportan en dev**. Sirve para comparar modelos; la evaluación con umbral fijo sobre test es de la Etapa 4.
- Se recomienda `tests/test_evaluate.py` con casos mínimos (un clasificador perfecto da PR-AUC = 1; uno aleatorio se acerca a la tasa de fraude; el costo se calcula a mano en un ejemplo de 10 filas).

### Acción 2. Registro de experimentos

| Qué hacer | Dónde | Qué se logra | Por qué |
|---|---|---|---|
| Función `log_result(row, path)` que agrega una fila a `reports/results.csv`. Esquema: `exp_id`, `timestamp`, `stage` (baseline / ablation / nn), `model`, `variant`, `cost_strategy`, `seed`, `hyperparams` (JSON), `n_features`, `best_iter`, `train_time_s`, `pr_auc_train`, `pr_auc`, `roc_auc`, `recall_at_p`, `f1_max`, `min_cost`, `notes` | `src/evaluate.py` + `reports/results.csv` | Una única fuente para la tabla comparativa del README, reproducible y sin copiar números a mano | Regla 7; además `pr_auc_train` frente a `pr_auc` permite detectar sobreajuste y deriva |

Las predicciones en dev de las configuraciones finales se guardan en `reports/preds/` (ignorado en git): se necesitan en la Etapa 4 para elegir el umbral, la calibración y las curvas.

### Acción 3. Baselines: regresión logística y XGBoost

| Qué hacer | Dónde | Qué se logra | Por qué |
|---|---|---|---|
| Entrenar con `X` (383 columnas): (a) clasificador sin información (PR-AUC = tasa de fraude); (b) regresión logística con `class_weight`, eligiendo `C` en dev; (c) XGBoost con `scale_pos_weight` = 27.43, `eval_set` = dev y early stopping sobre `aucpr`. Registrar tiempos y `pr_auc_train` | `notebooks/03_baselines.ipynb` (celdas: carga, pesos, logística, XGBoost, tabla comparativa) | Fija el número a superar y muestra cuánto aporta cada ingrediente (no linealidad, árboles) | **B1:** piso 0.0343. **G1:** relaciones no lineales, así que se espera ver logística < XGBoost. El pipeline ya dejó `X` sin NaN, escalada y sin columnas redundantes |

Notas: el número de árboles se elige con dev (early stopping), así que su PR-AUC en dev es algo optimista; para comparar, las redes se tratan igual. LightGBM es opcional si sobra tiempo.

### Acción 4. Ablaciones del importe y la imputación (heredadas de la Etapa 2)

| Qué hacer | Dónde | Qué se logra | Por qué |
|---|---|---|---|
| Correr con la misma logística y el mismo XGBoost, comparando PR-AUC en dev. **Importe:** (a) solo `amt_log`; (b) + `amt_cents`; (c) + `amt_freq`; (d) + `amt_price_point` (solo red con embeddings). **Imputación:** (a) mediana + indicador (por defecto); (b) 0 + indicador; (c) mediana sin indicador; (d) NaN nativos (solo XGBoost) | `notebooks/03_baselines.ipynb` (sección de ablaciones); filas con `stage = ablation` en `results.csv` | Decide qué variables y qué tratamiento de nulos se quedan, con evidencia | **C3** y **A10**: son las dos decisiones del preprocesamiento que más dependen del modelo |

Cómo se implementa cada variante:

- **Importe (a)-(c):** quitando columnas de `X` ya procesada. **Primero hay que comprobar en `models/preprocessor_meta.json` que `amt_cents` y `amt_freq` sobrevivieron al filtro de redundancia.** Si alguna fue eliminada, esa variante exige un `Preprocessor` aparte.
- **Imputación (b):** `numeric_impute: zero` con un `Preprocessor` nuevo ajustado con train y guardado en otra carpeta (regla 5). **(c):** quitar las columnas `null_pat_*` de `X`. Las dummies `missing` también codifican ausencia, así que el efecto será parcial y se anota. **(d):** requiere una variante del pipeline que deje los NaN (`xgb_native_nan`, hoy no implementada): de prioridad baja, solo si sobra tiempo.
- Una variante que no mejora en dev se elimina de la config. Si implica quitar columnas del pipeline, se vuelve a ajustar el `Preprocessor` y se corren de nuevo los tests anti-fuga.

### Acción 5. Ablación de `drop_candidates`

| Qué hacer | Dónde | Qué se logra | Por qué |
|---|---|---|---|
| Quitar de `X` las columnas derivadas de `card4`, `M5`, `id_23`, `id_27` e `id_34` (dummies `card4__*`, `M5__*`…) y comparar PR-AUC en dev con logística y XGBoost | `notebooks/03_baselines.ipynb` | Decide si se descartan de la config | **F3:** V de Cramér < 0.03; el EDA dejó la decisión pendiente para el desempeño en dev |

### Acción 6. Las cuatro arquitecturas

| Qué hacer | Dónde | Qué se logra | Por qué |
|---|---|---|---|
| Una función por arquitectura que **construye y no compila** (la pérdida depende de la estrategia de costo y la elige `train.py`): `build_mlp()` (ablación: Dense + ReLU, inicialización He, sin regularización), `build_mlp_regularizado()` (L2, Dropout, Batch Normalization), `build_mlp_embeddings()` (un `Embedding` por columna de `idx` con `input_dim` = `vocab_sizes`, concatenado con `X`) y `build_residual_mlp()` (proyección inicial y bloques Dense → BN → ReLU con conexión residual) | `src/models.py` (archivo nuevo) | Las cuatro arquitecturas del estudio comparativo, intercambiables y configurables desde la config | **Pregunta de investigación del README** y **F2:** la señal está en variables de alta cardinalidad. Salida con `sigmoid`; opcional: inicializar el sesgo de salida en `log(n1 / n0)` |

### Acción 7. Entrenamiento

| Qué hacer | Dónde | Qué se logra | Por qué |
|---|---|---|---|
| Una función `train_model(model, data, cfg, strategy, seed)`: fija la semilla; compila con entropía cruzada binaria + `class_weight` (de `compute_class_weights`) **o** focal loss (`BinaryFocalCrossentropy`, sin pesos de clase para no contarlos dos veces); early stopping sobre PR-AUC en dev con `restore_best_weights`; guarda historial y pesos en `models/` (ignorado en git); devuelve las predicciones en dev y registra la fila en `results.csv` | `src/train.py` (se añade a `compute_class_weights`) | Un único punto de entrenamiento, reproducible por semilla, con el mismo criterio de parada para todas las redes | **B1 / Acción 9 de la Etapa 2:** el desbalance se trata aquí. Con lotes de 2048 y 3.5% de fraude hay ~70 fraudes por lote, suficientes para estabilizar el gradiente |

### Acción 8. Experimentos de redes

| Qué hacer | Dónde | Qué se logra | Por qué |
|---|---|---|---|
| **Fase A:** búsqueda aleatoria de hiperparámetros con `seeds[0]` por arquitectura con `class_weight` (ancho, profundidad, Dropout, L2, tasa de aprendizaje, tamaño de lote, dimensión de embeddings); para `focal` se busca solo `(gamma, alpha)` con la mejor arquitectura. **Fase B:** con la mejor configuración de cada (arquitectura × estrategia), repetir con las 3 semillas | `notebooks/04_neural_networks.ipynb` (secciones: carga, búsqueda, semillas finales, curvas de entrenamiento) | Matriz final: 4 arquitecturas × 2 estrategias × 3 semillas (24 corridas) más las de búsqueda, todas registradas | Regla 8. El número de corridas se acota según el tiempo de CPU disponible |

### Acción 9. Consolidar resultados y figuras

| Qué hacer | Dónde | Qué se logra | Por qué |
|---|---|---|---|
| Tabla comparativa (media ± desviación en dev por modelo y estrategia) leída de `results.csv`; figuras: `20_ablation_importe_imputacion.png`, `21_pr_curves_dev.png` (baselines y mejores redes) y `22_training_curves.png` | `notebooks/04_neural_networks.ipynb` y `reports/figures/` | El material del avance y del README, sin copiar números a mano | La tabla comparativa del README se llena con esto (en dev); la versión final, con test, es de la Etapa 4 |

### Acción 10. Autoencoder (solo si hay tiempo)

| Qué hacer | Dónde | Qué se logra | Por qué |
|---|---|---|---|
| Entrenar con las transacciones legítimas de train; el puntaje de anomalía es el error de reconstrucción; evaluar PR-AUC en dev | `src/models.py`, `src/train.py` | Una comparación opcional con un enfoque no supervisado | Está marcado como opcional en el README y en el plan |

### Acción 11. Verificación y cierre

| Qué hacer | Dónde | Qué se logra | Por qué |
|---|---|---|---|
| Correr `python -m pytest`; revisar que `results.csv` no tenga filas incompletas; actualizar el estado y la tabla de resultados del README | tests, `reports/results.csv`, `README.md` | La etapa queda verificable y documentada | Regla 9; criterio "lista" de esta etapa |

---

## Entregables de la etapa

| Archivo | Qué hacer |
|---|---|
| `configs/default.yaml` | Acción 0 |
| `src/evaluate.py` | Acciones 1 y 2 (primera versión; se amplía en la Etapa 4) |
| `tests/test_evaluate.py` | Recomendado: pruebas pequeñas de las métricas |
| `src/models.py` | Acción 6 (y la 10 si hay tiempo) |
| `src/train.py` | Acción 7 (junto a `compute_class_weights`) |
| `notebooks/03_baselines.ipynb` | Acciones 3 a 5 |
| `notebooks/04_neural_networks.ipynb` | Acciones 8 y 9 |
| `reports/results.csv` | Una fila por experimento |
| `reports/figures/` | `20_ablation_importe_imputacion.png`, `21_pr_curves_dev.png`, `22_training_curves.png` |
| `README.md` | Estado de la etapa y tabla de resultados en dev |

---

## Decisiones abiertas (se resuelven al implementar)

1. Estrategia de costo para las redes: `class_weight` o `focal`, y los valores de `(gamma, alpha)`.
2. Espacio de búsqueda, número de pruebas y presupuesto de tiempo (CPU).
3. Backend de las redes (TensorFlow/Keras, como indica el plan) y su instalación en Python 3.12.
4. Dimensión de los embeddings (la regla propuesta es solo un punto de partida).
5. Qué variantes del importe y de la imputación sobreviven; si se implementa la variante de NaN nativos para XGBoost.
6. Si se descartan las `drop_candidates`.
7. Si se usa LightGBM además de XGBoost.
8. Si se compara `corr_threshold` (0.90 / 0.99) y si se prueba la familia de navegador sin versión (ambas cambian el pipeline).
9. Si se usa bootstrap sobre dev para decidir entre variantes casi empatadas.

## Fuera de alcance en esta etapa

- Usar el conjunto de test (Etapa 4).
- Elegir el umbral final, calibración, análisis de costo, perfil de falsos negativos y SHAP (Etapa 4).
- `docs/model_card.md` completo (Etapa 4; ahora solo notas).
- `requirements.txt` con versiones fijas (al final del proyecto).
- Arquitecturas secuenciales (LSTM): no hay identificador de cliente.