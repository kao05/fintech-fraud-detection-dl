# Como se manejara el proyecto

## Distribución de etapas

### Resumen

| Etapa | Objetivo | Cuándo | Está "lista" cuando... |
|---|---|---|---|
| 1. Datos y exploración | Entender el dataset | Primer avance | Tienes la tabla *hallazgo → decisión* |
| 2. Preparación y protocolo | Fijar las reglas del experimento | Semana 1 | El split y el pipeline existen y el test anti-fuga pasa |
| 3. Modelado | Entrenar baselines y redes | Semana 2 | `results.csv` tiene todos los experimentos |
| 4. Evaluación y publicación | Evaluar, interpretar y presentar | Semana 3 | El README cuenta el proyecto sin que nadie abra tus notebooks |

---

### Etapa 1 Datos y exploración

| Archivo | Qué hacer |
| :--- | :--- |
| `.gitignore` | Ignorar `data/raw/`, `data/processed/`, modelos pesados, `.venv` y credenciales de Kaggle. Es lo que ya haces |
| `data/README.md` | Fuente (enlace a la competencia), nota de que no se redistribuyen los datos, cómo descargarlos, qué archivos se usan (solo `train_*`) y por qué (el test no tiene etiquetas), forma esperada (`590540, 434`) y un **diccionario de familias de columnas** (`C`, `D`, `M`, `V`, `id`, `card`...) |
| `src/data.py` | Tu código de carga y unión convertido en funciones: `load_raw()`, `merge_tables()`, `save_processed()` y `load_processed()`. Así cada notebook carga los datos en una línea. Aprovecha para mejorarlo: rutas relativas a la raíz del proyecto (no `"../data/raw/"`, que depende de dónde ejecutes), `float64` $\rightarrow$ `float32`, ordenar por `TransactionDT` y una columna `has_identity` |
| `notebooks/01_eda.ipynb` | Estructura, objetivo, importes, categóricas, faltantes informativos, correlaciones con el objetivo, deriva temporal. Cierra con la **tabla de decisiones** |
| `reports/figures/` | Las gráficas del EDA, guardadas con nombres descriptivos (`01_class_balance.png`) para reutilizarlas en el avance y el README |
| `requirements.txt` | Versión inicial: pandas, pyarrow, matplotlib, seaborn, scipy, jupyter |

---

### Etapa 2 Preparación y protocolo experimental

| Archivo | Qué hacer |
| :--- | :--- |
| `configs/default.yaml` | Semillas (p. ej. 42, 1, 2), fracciones del split (0.70 / 0.15 / 0.15), columnas a descartar, listas de categóricas, supuestos de costo (`c_FN`, `c_FP`) y la precisión objetivo. Regla: **nada de esto va escrito a mano en los notebooks** |
| `src/split.py` | Una función de **split temporal**: ordena por `TransactionDT`, corta en train/dev/test y verifica que los tiempos no se solapen. Es la *única* definición del split del proyecto: todos los notebooks la llaman |
| `src/preprocessing.py` | El pipeline, **ajustado solo con train** y aplicado a dev/test: descartar `TransactionID` y `TransactionDT` crudo, derivar hora y día, indicadores de nulo, imputación, codificación de categóricas (índices enteros para embeddings, frecuencias o one-hot para las demás), escalado de numéricas y `log1p` del importe. Guarda el transformador ajustado para reutilizarlo. Las categorías nuevas en dev/test van a un índice “desconocido” |
| `notebooks/02_preprocessing_split.ipynb` | Muestra el proceso: reporta tamaño y tasa de fraude de cada split, aplica el pipeline y guarda los datos listos. Aquí se ve si el fraude cambia con el tiempo |
| `tests/test_no_leakage.py` | Comprobaciones automáticas: `max(t_train) < min(t_dev)` y `max(t_dev) < min(t_test)`, sin `TransactionID` repetidos entre splits y estadísticas del escalador iguales a las de train. Usa datos sintéticos pequeños para que corra en segundos |

---

### Etapa 3 Modelado

| Archivo | Qué hacer |
| :--- | :--- |
| `src/evaluate.py` (primera versión) | Funciones de métricas: PR-AUC, ROC-AUC, precision, recall, F1 y recall a precisión fija. Se necesita ya para comparar los baselines |
| `notebooks/03_baselines.ipynb` | Regresión logística y XGBoost/LightGBM. Fijan el número a superar y muestran cuánto aporta la red |
| `src/models.py` | Una función por arquitectura: `build_mlp()` (ablación), `build_mlp_regularizado()`, `build_mlp_embeddings()` y `build_residual_mlp()`. Cada una devuelve un modelo de Keras listo para entrenar |
| `src/train.py` | Una función de entrenamiento que recibe modelo, datos y configuración: fija la semilla, aplica pesos de clase o focal loss, usa early stopping sobre dev, y guarda historial y pesos |
| `notebooks/04_neural_networks.ipynb` | Orquesta los experimentos: arquitectura × estrategia de costo × semillas. Primero ajusta con una semilla (búsqueda aleatoria de hiperparámetros) y, con la mejor configuración, repite con 3 semillas |
| `reports/results.csv` | Una fila por experimento: modelo, estrategia, semilla, hiperparámetros y métricas en dev. Es la fuente de la tabla comparativa |

---

### Etapa 4 Evaluación, interpretación y publicación

| Archivo | Qué hacer |
| :--- | :--- |
| `src/evaluate.py` (ampliado) | Selección de umbral en dev, costo esperado, curva PR y calibración (Brier y curva de confiabilidad) |
| `notebooks/05_evaluation.ipynb` | **Una sola** evaluación en test con el umbral ya elegido. Después: media ± desviación entre semillas, análisis de costo, perfil de los falsos negativos (¿qué fraudes se escapan?), importancia de variables (SHAP o permutación) y comparación honesta contra XGBoost |
| `docs/model_card.md` | Uso previsto, datos, métricas, limitaciones (variables anonimizadas, deriva temporal, solo comercio electrónico sin tarjeta presente) y riesgos |
| `README.md` | Problema, pregunta, dataset, metodología, **tabla comparativa**, 2-3 gráficas clave, conclusiones y cómo reproducir (pasos exactos, desde clonar hasta ejecutar) |
| `requirements.txt` (final) | Versiones fijas, generadas con `pip freeze` al terminar |

---

### Metricas propuestas 

| Métrica | ¿Qué nos permite evaluar? |
| :--- | :--- |
| Precision | Cuántas alertas generadas corresponden realmente a fraude. |
| Recall | Qué proporción del fraude conseguimos detectar. |
| F1-score | Equilibrio entre precision y recall. |
| PR-AUC | Calidad de la detección considerando el desbalance. |
| ROC-AUC | Capacidad de discriminación entre clases. |
| Matriz de confusión | Distribución de aciertos y errores. |

----

| Carpeta | Responsabilidad |
|---|---|
| `src/` | Lógica reutilizable: funciones que se llaman desde cualquier notebook |
| `notebooks/` | La narrativa: ejecutan esas funciones, muestran gráficas y explican decisiones |
| `configs/` | Parámetros: semillas, fracciones del split, hiperparámetros, supuestos de costo |
| `tests/` | Garantías: comprueban que no hay fuga de datos |
| `reports/` y `docs/` | Salidas: figuras, tabla de resultados, model card |


