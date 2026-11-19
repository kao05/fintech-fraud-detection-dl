# Deep Learning-Based Financial Transaction Fraud Detection: A Comparative Study of Neural Network Architectures

**Problema a solucionar**
Al realizar una transacción una fracción pequeña de transacciones es fraudulenta, pero cada una es costosa, y bloquear transacciones legítimas daña la experiencia del cliente.

**Objetivo general**
Desarrollar y evaluar modelos de redes neuronales profundas capaces de clasificar transacciones financieras como legítimas o fraudulentas, comparando distintas arquitecturas y estrategias de entrenamiento bajo condiciones de desbalance de clases.

**Pregunta a resolver**
`¿En qué medida las arquitecturas MLP regularizadas y residuales mejoran la detección de transacciones fraudulentas frente a una MLP convencional, considerando el desbalance de clases y el compromiso entre la detección de fraude y las falsas alertas?`

**Hipotesis**
Las arquitecturas neuronales con mayor capacidad de representación y mecanismos de regularización podrían capturar relaciones no lineales entre las características transaccionales y mejorar la detección de fraude frente a una MLP básica.

## Redes Neuronales con las cuales experimentaremos (de inicio)

### Modelo 1. MLP convencional
Una red neuronal densa con varias capas ocultas.
- Dense + ReLU.
- Inicialización He.
- Entropía cruzada binaria.
- Optimizador Adam.

### Modelo 2. MLP con regularización
Una arquitectura densa que incorpore mecanismos para mejorar la generalización.
- L2.
- Dropout.
- Batch Normalization.
- Early Stopping.

### Modelo 3. Residual MLP
Una red neuronal densa que incorpore conexiones residuales (skip connections).
- Bloques densos.
- Batch Normalization.
- ReLU.
- Conexiones residuales.
- Optimizador Adam.

### Entrenamiento sensible al costo mediante pesos de clase 

Para poder evitar que aunque prediga muy bien la clase legitima evite la clase de fraude.


**Como dato importante:** Si el dataset hubiera contado con informacion acerca de las horas de las transacciones se pudo haber realizado una arquitectura `LSTM`.


## Baseline 

### Regresión logística


### XGBoost



### Autoencoders (Si es que nos da tiempo )



## Arquitectura del proyecto 

```
fintech-fraud-detection/
├── README.md                # problema, resultados, tabla comparativa, cómo reproducir
├── requirements.txt         # con versiones fijas
├── .gitignore               # data/, modelos pesados
├── configs/default.yaml     # semillas, fracciones del split, hiperparámetros
├── data/
│   ├── README.md            # 
│   ├── raw/                 # ignorado
│   └── processed/           # Parquet, ignorado
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_split.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_neural_networks.ipynb
│   └── 05_evaluation.ipynb
├── src/                     # data.py, split.py, preprocessing.py, models.py, train.py, evaluate.py
├── tests/test_no_leakage.py # comprueba que max(t_train) < min(t_dev) < min(t_test)
├── reports/                 # figures/, results.csv
└── docs/model_card.md
```