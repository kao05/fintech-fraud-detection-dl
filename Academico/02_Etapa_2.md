# Qué se hará específicamente en la Etapa 2: Preparación y protocolo experimental

**Objetivo:** fijar las reglas del experimento. Un split temporal único y un pipeline de preprocesamiento ajustado solo con train, guardado y verificado con un test anti-fuga.

**La etapa está "lista" cuando:** el split y el pipeline existen, `tests/test_no_leakage.py` pasa, `data/processed/` contiene train/dev/test listos y el notebook 02 reporta el tamaño y la tasa de fraude de cada split.

---

## Estado de partida (lo que ya existe)

- `src/data.py`: `load_raw()`, `merge_tables()`, `save_processed()`, `load_processed()` y `column_groups()`. Los datos ya están en `data/processed/train_merged.parquet`, ordenados por `TransactionDT`, con `float32` y la columna `has_identity`.
- `notebooks/01_eda.ipynb` completo, figuras `01` a `16` en `reports/figures/` y tabla *hallazgo → decisión* cerrada.
- Datos: 590,540 × 435 (434 originales + `has_identity`); 20,663 fraudes (3.5%); 394 numéricas, 31 categóricas de texto y 6 categóricas codificadas (`card1-3`, `card5`, `addr1-2`).

---

## Hallazgos del EDA que gobiernan esta etapa

| Código | Hallazgo | Implicación |
|---|---|---|
| A2 | El 24.4% de las transacciones tiene registro de identidad; los nulos de `id` y `device` son sobre todo estructurales | Conservar `has_identity`; no interpretar esos nulos como ausencia aleatoria |
| A3 | Nulos medios por familia: `id` 84.8%, `device` 78.0%, `dist` 76.6%, `D` 58.2%, `M` 49.9%, `email` 46.4%, `V` 43.0%, `addr` 11.1%, `card` 0.5%, `C` 0.0% | Tratamiento por familia, no una sola regla global |
| A4 / A7 | `TransactionDT` casi único por fila; `TransactionID` único | Ninguno de los dos entra crudo como predictor |
| A5 | `card1` tiene ~13.5k valores únicos y es un código categórico | No usar one-hot; embeddings o frecuencia |
| A8 | 3 pares de filas idénticas, todas legítimas | No deduplicar |
| A9 | Sin columnas constantes; 93 numéricas con ≤10 valores únicos (92 `V` y 1 `id`) | Esas siguen siendo numéricas (banderas y conteos) |
| A10 | 227 de 394 numéricas tienen mediana 0. En las columnas con mayor diferencia, el fraude cuando el valor falta es ~2.1-2.7% y cuando vale 0 es 8-21% (p. ej. `D7`: 2.70% vs 21.16%) | El nulo no equivale a 0: indicador de nulo obligatorio y **ningún nulo se imputa sin indicador** |
| B1 | 3.5% de fraude, ratio 27.6 : 1 | Pesos de clase o focal loss en entrenamiento; nada de sobremuestreo en dev/test |
| C1 / C2 | Importe de cola muy larga; fraude y legítimas con medianas similares | `log1p` y no eliminar outliers |
| C3 | El valor exacto del importe importa: algunos precios recurrentes tienen tasas muy distintas de la base de 3.5% (57.95 → 0.38%, 47.95 → 0.39%, 35.95 → 0.29%, 150.00 → 6.71%), mientras que "redondo vs no redondo" casi no separa (3.59% vs 3.41%) | Derivar `amt_cents`, `amt_freq` y `amt_price_point`; **no** crear `amt_is_round`; guardar el importe crudo en `meta` |
| D1 | Tasa de fraude no estacionaria (~2-3% → ~4-5%) con picos de volumen | Split temporal obligatorio |
| D2 | Pico de fraude en la hora relativa 7 (~10.6%), mínimo en 13-14 (~2.3%) | Derivar hora con codificación cíclica |
| F1 / F2 / F4 | Señal categórica concentrada en alta cardinalidad; `<NaN>` de `DeviceType`, `DeviceInfo` e `id_31` equivale a "sin identidad" | One-hot para baja cardinalidad, índices y frecuencia para alta, y nulo como categoría explícita |
| F3 | V de Cramér < 0.03 en `card4`, `M5`, `id_23`, `id_27`, `id_34` | Candidatas a descartar, sin decidirlo todavía |
| G1 | |Spearman| máx ≈ 0.40; varias `V` con Spearman ≈ 0.3 y Pearson ≈ 0.02 | Relaciones no lineales; `log1p` en las no negativas antes de estandarizar |
| G2 | 1,443 pares con |r| > 0.9, el 97% entre `V` | Filtro de redundancia calculado solo con train |
| H2 / H3 | Dentro de identidad, la ausencia de `id_14`, `id_30`, `id_32`, `id_33`, `id_34` e `id_03`, `id_04`, `id_09`, `id_10` distingue clases; solo hay 69 patrones distintos de nulos | Un indicador de nulo por patrón (≤ 69), no por columna |

---

## Reglas que no se rompen

1. Todo estadístico (medianas, media, desviación, vocabularios, frecuencias, patrones de nulos, correlaciones) se calcula **solo con train** y se aplica a dev y test.
2. El split se define **una sola vez**, en `src/split.py`, y se hace **antes** de ajustar cualquier transformación.
3. No se sobremuestrea dev ni test.
4. No se construyen variables con información futura ni agregados calculados sobre el dataset completo.
5. Nada va escrito a mano en los notebooks: todo sale de `configs/default.yaml`.
6. Se elige modelo con dev, nunca con test. El test no se toca en esta etapa.
7. Ningún valor faltante se imputa sin su indicador de nulo: el nulo no equivale a 0 (hallazgo A10).

---

## Orden del pipeline

| # | Acción | Dónde | ¿Se ajusta con train? |
|---|---|---|---|
| 0 | Configuración centralizada | `configs/default.yaml` | No |
| 1 | Split temporal | `src/split.py` | No (usa solo el tiempo) |
| 2 | Limpieza: descartar columnas no predictoras | `src/preprocessing.py` | No |
| 3 | Variables derivadas: hora cíclica, `log1p` del importe y centavos | `src/preprocessing.py` | No |
| 4 | Indicadores de nulo por patrón | `src/preprocessing.py` | **Sí** |
| 5 | Imputación | `src/preprocessing.py` | **Sí** |
| 6 | Codificación de categóricas | `src/preprocessing.py` | **Sí** |
| 7 | Transformación de colas y escalado | `src/preprocessing.py` | **Sí** |
| 8 | Filtro de redundancia | `src/preprocessing.py` | **Sí** |
| 9 | Desbalance: decisión de diseño | `src/train.py` (Etapa 3) | n/a |
| 10 | Ajustar, guardar y aplicar el pipeline | `src/preprocessing.py` + notebook 02 | **Sí** |
| 11 | Test anti-fuga | `tests/test_no_leakage.py` | n/a |

---

## Acciones

### Acción 0. Configuración centralizada

| Qué hacer | Dónde | Qué se logra | Por qué (EDA) |
|---|---|---|---|
| Crear el archivo con semillas, fracciones del split, listas de columnas, umbrales y supuestos de costo | `configs/default.yaml` | Una única fuente de verdad: cambiar un parámetro no exige tocar código ni notebooks | La tabla de decisiones produjo listas y umbrales (`cat_cols`, `num_cols`, `drop_columns`, `corr_threshold`) que no deben vivir en celdas sueltas |

Esqueleto propuesto (los valores marcados con `?` se deciden al implementar):

```yaml
seeds: [19, 69, 79]
split: {train: 0.70, dev: 0.15, test: 0.15}
columns:
  target: isFraud
  id: TransactionID
  time: TransactionDT
  drop: [TransactionID, TransactionDT]      # TransactionDT crudo se descarta tras derivar la hora
  categorical_numeric_coded: [card1, card2, card3, card5, addr1, addr2]
  drop_candidates: [card4, M5, id_23, id_27, id_34]   # solo candidatas: no se descartan aquí
preprocessing:
  onehot_max_cardinality: 10        # ?
  min_category_freq: 50             # ?
  skew_threshold: 2.0               # ?
  corr_threshold: 0.95              # ?
  amt_price_point_min_freq: ?       # frecuencia mínima (en train) para que un importe exacto tenga índice propio
  add_null_indicators: true         # obligatorio: ningún nulo se imputa sin indicador
  numeric_impute: median
  xgb_native_nan: false             # variante opcional: dejar NaN sin imputar para XGBoost
cost: {c_FN: ?, c_FP: ?, target_precision: ?}
```

### Acción 1. Split temporal

| Qué hacer | Dónde | Qué se logra | Por qué (EDA) |
|---|---|---|---|
| Ordenar por `TransactionDT`, cortar 70 / 15 / 15 **por valor de tiempo** y verificar que no se solapen. Función única que todos los notebooks llaman | `src/split.py` | dev y test simulan el futuro real: se entrena con el pasado y se evalúa con lo posterior | **D1:** la tasa de fraude cambia con el tiempo (~2-3% → ~4-5%); un split aleatorio mezclaría épocas y daría métricas optimistas. **A8:** hay filas con el mismo `TransactionDT`; el corte por valor de tiempo evita que un empate rompa `max(t_train) < min(t_dev)` |

Salidas del notebook 02: tamaño y tasa de fraude de cada split, y una figura con la tasa diaria coloreada por split.

### Acción 2. Limpieza: descartar columnas no predictoras

| Qué hacer | Dónde | Qué se logra | Por qué (EDA) |
|---|---|---|---|
| Descartar `TransactionID` y `TransactionDT` crudo (después de derivar la hora). Conservar `has_identity`. **No** deduplicar filas | `src/preprocessing.py` | Se evita que el modelo memorice identificadores o un contador de tiempo que dev/test nunca tendrán en el mismo rango | **A7 / A4:** ambos son casi únicos por fila. **A8:** los 3 pares duplicados son cargos legítimos y su efecto es despreciable. **H1:** `has_identity` triplica la tasa de fraude |

### Acción 3. Variables derivadas

| Qué hacer | Dónde | Qué se logra | Por qué (EDA) |
|---|---|---|---|
| `hour = (TransactionDT // 3600) % 24`, codificada como `hour_sin` y `hour_cos`. `amt_log = log1p(TransactionAmt)` y `amt_cents` (centavos, 0-99, calculados en `float64`). El importe crudo no entra a `X`: se guarda en `meta` | `src/preprocessing.py` | La red recibe el patrón horario como variable continua y cíclica, un importe sin cola larga y los centavos como señal propia que un modelo suave sobre `log1p` no ve | **D2:** el fraude llega a ~10.6% en la hora 7 y baja a ~2.3% hacia la 13. **C1 / C2:** el importe tiene cola muy larga y no se eliminan outliers, así que `log1p` los absorbe. **C3:** los precios recurrentes tienen tasas de fraude muy distintas (p. ej. 57.95 → 0.38%) y el importe redondo por sí solo no separa (3.59% vs 3.41%), por eso no se crea `amt_is_round` |

Notas: la hora es **relativa** (el desfase de `TransactionDT` es desconocido), y eso debe quedar en la model card. **No** se usa el índice de día absoluto como variable (crece con el tiempo y dev/test caerían fuera del rango visto). Si se quisiera un "día de la semana" relativo (`day % 7`), se prueba en dev y no se asume.

El importe crudo no entra a `X`: XGBoost ve `amt_log` igual que el crudo (los árboles son invariantes a transformaciones monótonas), y el crudo se guarda en `meta` para el costo ponderado por importe, el análisis de falsos negativos y las gráficas en dólares de la Etapa 4.

### Acción 4. Indicadores de nulo por patrón

| Qué hacer | Dónde | Qué se logra | Por qué (EDA) |
|---|---|---|---|
| Con train, agrupar las columnas con nulos por patrón idéntico de ausencia y crear **un indicador por patrón** (≤ 69). Aplicar los mismos grupos a dev/test | `src/preprocessing.py` (clase de indicadores con `fit` y `transform`) | Se conserva la información de "qué falta" sin añadir cientos de columnas casi duplicadas | **H3:** solo hay 69 patrones distintos. **H2:** dentro de identidad, la ausencia de varias `id_*` distingue fraude. **A2:** buena parte de los nulos es estructural, y los indicadores permiten al modelo distinguirlo |

### Acción 5. Imputación

| Qué hacer | Dónde | Qué se logra | Por qué (EDA) |
|---|---|---|---|
| **Primero** los indicadores de la acción 4; **después** imputar. Numéricas: mediana de train (en las 227 columnas con mediana 0 equivale a 0, y el indicador es lo que distingue "falta" de "vale 0"). Categóricas: categoría explícita `"missing"`. Ningún nulo se imputa sin indicador. Variante opcional para XGBoost: dejar los NaN sin imputar | `src/preprocessing.py` | Matriz sin NaN (las redes no aceptan nulos) sin borrar el rastro de la ausencia, y una variante donde XGBoost aprende por sí solo la dirección de los nulos | **A3:** `C` y `card` casi no tienen nulos; `id`, `device`, `dist`, `D`, `M`, `email` y `V` sí. **F4:** el `<NaN>` de `DeviceType`, `DeviceInfo` e `id_31` equivale a "sin identidad" (~2% de fraude), así que es información y no ruido. **A10:** en `D7` el fraude es 2.70% cuando falta y 21.16% cuando vale 0; imputar 0 sin indicador mezclaría ambos grupos |

Nota: con el indicador presente, para un modelo lineal la constante imputada es irrelevante (se absorbe en el coeficiente del indicador) y para la MLP el efecto es pequeño. Sin el indicador sí se perdería la señal. Por eso el indicador es obligatorio y la constante se decide con una ablación en dev.

### Acción 6. Codificación de categóricas

| Qué hacer | Dónde | Qué se logra | Por qué (EDA) |
|---|---|---|---|
| **Baja cardinalidad** (≤ `onehot_max_cardinality`, p. ej. `ProductCD`, `card4`, `card6`, `DeviceType`, `M1`-`M9`): one-hot. **Alta cardinalidad** (`card1`, `card2`, `card3`, `addr1`, `DeviceInfo`, `id_31`, correos...): (a) índice entero para embeddings y (b) frecuencia de train para modelos tabulares. Categorías con frecuencia < `min_category_freq` o nuevas en dev/test van al índice **"desconocido"**. Vocabularios y frecuencias salen **solo de train** | `src/preprocessing.py` (clase codificadora) | Las categóricas de mucha señal entran sin explotar la dimensionalidad, y dev/test no rompen el pipeline con valores nunca vistos | **F2 / A5:** la mayor asociación con el fraude está en `card1` (V de Cramér 0.32), `DeviceInfo` (0.27) y `card2` (0.23), todas de alta cardinalidad. **F1:** `ProductCD`, `card6` y `DeviceType` tienen tasas muy distintas de la base y pocas categorías |

Importe exacto como categórica de alta cardinalidad (hallazgo **C3**): `amt_freq` (frecuencia del valor exacto en train; valores nuevos → 0) y `amt_price_point` (índice del valor exacto si tiene frecuencia ≥ `amt_price_point_min_freq` en train; el resto y los nuevos → "otros"). **Qué se logra:** el modelo puede aprender el efecto de los precios recurrentes (p. ej. 57.95 con 0.38% de fraude frente a 150.00 con 6.71%), que un importe continuo suavizado no captura. Ambos salen solo de train.

### Acción 7. Transformación de colas y escalado

| Qué hacer | Dónde | Qué se logra | Por qué (EDA) |
|---|---|---|---|
| En numéricas con asimetría alta en train (`skew_threshold`) y mínimo ≥ 0, aplicar `log1p` (con recorte a 0 en dev/test). Luego estandarizar con **media y desviación de train**. Las banderas binarias (2 valores) no se escalan | `src/preprocessing.py` | Escalas comparables y distribuciones menos sesgadas: el optimizador converge mejor y ninguna variable domina por su magnitud | **G1:** varias `V` tienen Spearman ≈ 0.3 con Pearson ≈ 0.02 (relación monótona pero con colas), y **C2** muestra colas largas. **A9 / E1:** muchas `V` son banderas 0/1 o conteos pequeños que no necesitan escalado |

### Acción 8. Filtro de redundancia

| Qué hacer | Dónde | Qué se logra | Por qué (EDA) |
|---|---|---|---|
| Calcular correlaciones **solo con train**; de cada par con \|r\| > `corr_threshold` (propuesta 0.95), eliminar una columna (la de menor asociación con el objetivo en train). Reportar cuántas columnas elimina cada umbral (0.90, 0.95, 0.99). **No** descartar aún `drop_candidates` | `src/preprocessing.py` (clase de filtro) + notebook 02 | Menos columnas redundantes: entrenamiento más rápido, menos riesgo de sobreajuste y una logística más estable | **G2:** 1,443 pares con \|r\| > 0.9, 97% entre `V`, con bloques de correlación 0.8-0.99. **F3:** las candidatas se confirman con desempeño en dev (Etapa 3), no por V de Cramér |

### Acción 9. Desbalance de clases (decisión de diseño)

| Qué hacer | Dónde | Qué se logra | Por qué (EDA) |
|---|---|---|---|
| **No** remuestrear en el preprocesamiento. Los pesos de clase (calculados con la proporción de train) o focal loss se aplican en el entrenamiento. Dejar la estrategia en la config | `configs/default.yaml` y `src/train.py` (Etapa 3) | Dev y test conservan la distribución real, así que las métricas reflejan el uso en producción | **B1:** 27.6 : 1 con 3.5% de fraude. El piso de PR-AUC es ≈ 0.035 en un conjunto con esa distribución, y solo tiene sentido compararlo con ella |

### Acción 10. Ajustar, guardar y aplicar el pipeline

| Qué hacer | Dónde | Qué se logra | Por qué (EDA) |
|---|---|---|---|
| Ajustar el pipeline con train; transformar dev y test; guardar el transformador ajustado; escribir los conjuntos listos | `src/preprocessing.py` (`fit`, `transform`, `save`, `load`); notebook 02 lo ejecuta y lo muestra | Cualquier notebook posterior carga datos listos en una línea y el mismo transformador se reutiliza sin reajustarlo | Garantiza la regla central de las etapas: todo estadístico viene de train |

Salidas propuestas:

| Archivo | Contenido |
|---|---|
| `data/processed/{train,dev,test}_X.parquet` | Numéricas procesadas, indicadores de nulo, one-hot y frecuencias |
| `data/processed/{train,dev,test}_idx.parquet` | Índices enteros de las categóricas de alta cardinalidad (para embeddings) |
| `data/processed/{train,dev,test}_y.parquet` | `isFraud` |
| `data/processed/{train,dev,test}_meta.parquet` | `TransactionID`, `TransactionDT` y `TransactionAmt` crudo (no son variables; sirven para el test anti-fuga, el costo ponderado por importe y el análisis de falsos negativos de la Etapa 4) |
| `models/preprocessor.joblib` + metadatos | Transformador ajustado y tamaños de vocabulario (ignorado en git) |

### Acción 11. Test anti-fuga

| Qué hacer | Dónde | Qué se logra | Por qué (EDA) |
|---|---|---|---|
| Pruebas automáticas con datos sintéticos pequeños (corren en segundos): `max(t_train) < min(t_dev)` y `max(t_dev) < min(t_test)`; sin `TransactionID` repetidos entre splits; media y desviación del escalador, medianas, vocabularios y patrones de nulos iguales a los calculados con train; sin NaN ni infinitos tras transformar; mismas columnas y orden en los tres splits | `tests/test_no_leakage.py` | Una garantía automática de que el protocolo se cumple y de que ningún cambio futuro introduce fuga sin que salte una alarma | **D1:** con deriva temporal, cualquier fuga infla las métricas. **A8:** los empates de `TransactionDT` exigen que el test del tiempo use el corte por valor |

---

## Entregables de la etapa

| Archivo | Qué hacer |
|---|---|
| `configs/default.yaml` | Acción 0 |
| `src/split.py` | Acción 1 |
| `src/preprocessing.py` | Acciones 2 a 8 y 10. Piezas sugeridas: `add_derived_features`, indicadores de nulo, imputador, codificador de categóricas, transformador numérico (`log1p` y escalado), filtro de redundancia y un `Preprocessor` que las orquesta |
| `notebooks/02_preprocessing_split.ipynb` | Muestra el proceso: tamaño y tasa de fraude por split, columnas antes y después de cada paso, aplica el pipeline y guarda los datos |
| `tests/test_no_leakage.py` | Acción 11 |
| `data/processed/` | Salidas de la acción 10 |
| `reports/figures/` | `17_split_timeline.png` (tasa de fraude diaria coloreada por split), `18_fraud_rate_by_split.png` (tamaño y tasa por split) y `19_corr_filter_threshold.png` (columnas eliminadas según el umbral) |
| `docs/model_card.md` | Solo notas: la hora es relativa, deriva temporal, `has_identity` como artefacto de recolección y que los precios recurrentes pueden cambiar con el tiempo. El documento completo es de la Etapa 4 |

---

## Decisiones abiertas (se resuelven al implementar)

1. `onehot_max_cardinality` y `min_category_freq`: fijarlos mirando `nunique` y las frecuencias en train.
2. `corr_threshold`: elegirlo con el reporte de columnas eliminadas (0.90, 0.95, 0.99).
3. `skew_threshold` y qué columnas reciben `log1p`.
4. Supuestos de costo (`c_FN`, `c_FP`) y la precisión objetivo.
5. Si se prueba un "día de la semana" relativo (`day % 7`).
6. `amt_price_point_min_freq`: fijarlo viendo cuántos valores exactos de importe superan cada frecuencia en train.
7. Qué variantes del importe y de la imputación se comparan en dev (ver la sección de ablaciones).

## Ablaciones que hereda la Etapa 3

Se corren en dev con la misma regresión logística y XGBoost, comparando PR-AUC:

| Tema | Variantes |
|---|---|
| Importe | (a) solo `amt_log`; (b) + `amt_cents`; (c) + `amt_freq`; (d) + `amt_price_point` (solo red con embeddings) |
| Imputación | (a) mediana + indicador (por defecto); (b) 0 + indicador; (c) mediana sin indicador; (d) NaN nativos (solo XGBoost) |

La variante que no mejore en dev se elimina de `configs/default.yaml`.

## Fuera de alcance en esta etapa

- Entrenar modelos (baselines y redes son de las Etapas 3 y 4).
- Usar el conjunto de test.
- `requirements.txt` (se pospone para el final).
