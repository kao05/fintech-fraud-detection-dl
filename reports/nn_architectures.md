# Arquitecturas de red

Valores de `configs/default.yaml` (más `OVERRIDES` si se usaron). Generado desde `04_neural_networks.ipynb`.

## Resumen comparativo

| Arquitectura | Entradas | Dense | BatchNorm | Dropout | Embedding | Add (residual) | Entrenables | No entrenables (BN) | Total | MB (float32) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. MLP convencional | 1 | 3 | 0 | 0 | 0 | 0 | 131329 | 0 | 131329 | 0.53 |
| 2. MLP regularizada | 1 | 3 | 2 | 2 | 0 | 0 | 131713 | 768 | 132481 | 0.53 |
| 3. MLP con embeddings | 14 | 3 | 2 | 2 | 13 | 0 | 254562 | 768 | 255330 | 1.02 |
| 4. Residual MLP | 1 | 8 | 7 | 3 | 0 | 3 | 495105 | 3584 | 498689 | 1.99 |

## Hiperparámetros

| Arquitectura | Hiperparámetros |
|---|---|
| 1. MLP convencional | {"hidden": [256, 128]} |
| 2. MLP regularizada | {"hidden": [256, 128], "dropout": 0.3, "l2": 1e-05, "batch_norm": true} |
| 3. MLP con embeddings | {"hidden": [256, 128], "dropout": 0.3, "l2": 1e-05, "batch_norm": true} |
| 4. Residual MLP | {"width": 256, "n_blocks": 3, "dropout": 0.2, "l2": 1e-05} |

## Protocolo de entrenamiento

| Elemento | Valor |
|---|---|
| Datos de entrada | X: 383 columnas; idx: 13 columnas de índices (solo la MLP con embeddings) |
| Filas | train 413,378 \| dev 88,581 (el test no se usa en esta etapa) |
| Optimizador | Adam, tasa de aprendizaje 0.001 |
| Tamaño de lote / épocas máximas | 2048 / 100 |
| Early stopping | PR-AUC exacto en dev, paciencia 10, restore_best_weights=True |
| Inicialización | He normal en las capas Dense; Embedding con la inicialización por defecto de Keras |
| Estrategia class_weight | entropía cruzada binaria con pesos de train 0.518 / 14.217; sesgo de salida inicial 0 |
| Estrategia focal | BinaryFocalCrossentropy (gamma=2.0, alpha=None), sin pesos de clase; sesgo de salida inicial log(n1/n0) = -3.312 |
| Costo asumido | c_FN=10, c_FP=1, precisión objetivo 0.5 |
| Semillas | [19, 69, 79] (búsqueda con 19) |
| Entorno | TensorFlow 2.21.0, Keras 3.15.1, 6 hilos de CPU |

## Embeddings

| Variable | Vocabulario | Dimensión | Parámetros | % de los parámetros de embeddings |
|---|---|---|---|---|
| card1_idx | 946 | 32 | 30272 | 48.7 |
| card2_idx | 445 | 32 | 14240 | 22.9 |
| card3_idx | 29 | 11 | 319 | 0.5 |
| card5_idx | 45 | 13 | 585 | 0.9 |
| addr1_idx | 72 | 18 | 1296 | 2.1 |
| addr2_idx | 7 | 5 | 35 | 0.1 |
| P_emaildomain_idx | 54 | 15 | 810 | 1.3 |
| R_emaildomain_idx | 35 | 12 | 420 | 0.7 |
| id_30_idx | 58 | 16 | 928 | 1.5 |
| id_31_idx | 65 | 17 | 1105 | 1.8 |
| id_33_idx | 45 | 13 | 585 | 0.9 |
| DeviceInfo_idx | 102 | 21 | 2142 | 3.4 |
| amt_price_point_idx | 295 | 32 | 9440 | 15.2 |

## 1. MLP convencional

| # | Capa | Tipo | Salida | Parámetros | Detalle |
|---|---|---|---|---|---|
| 0 | x | InputLayer | (383,) | 0 |  |
| 1 | dense | Dense | (256,) | 98304 | unidades=256, activación=relu, sesgo=True, inicialización=HeNormal |
| 2 | dense_1 | Dense | (128,) | 32896 | unidades=128, activación=relu, sesgo=True, inicialización=HeNormal |
| 3 | fraud_prob | Dense | (1,) | 129 | unidades=1, activación=sigmoid, sesgo=True, inicialización=GlorotUniform |

```text
Model: "mlp"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━
┃ Layer (type)                                        ┃ Output Shape                           ┃               Para
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━
│ x (InputLayer)                                      │ (None, 383)                            │                   
├─────────────────────────────────────────────────────┼────────────────────────────────────────┼───────────────────
│ dense (Dense)                                       │ (None, 256)                            │                98,
├─────────────────────────────────────────────────────┼────────────────────────────────────────┼───────────────────
│ dense_1 (Dense)                                     │ (None, 128)                            │                32,
├─────────────────────────────────────────────────────┼────────────────────────────────────────┼───────────────────
│ fraud_prob (Dense)                                  │ (None, 1)                              │                   
└─────────────────────────────────────────────────────┴────────────────────────────────────────┴───────────────────
 Total params: 131,329 (513.00 KB)
 Trainable params: 131,329 (513.00 KB)
 Non-trainable params: 0 (0.00 B)

```

## 2. MLP regularizada

| # | Capa | Tipo | Salida | Parámetros | Detalle |
|---|---|---|---|---|---|
| 0 | x | InputLayer | (383,) | 0 |  |
| 1 | dense | Dense | (256,) | 98048 | unidades=256, activación=linear, sesgo=False, inicialización=HeNormal, L2=1e-05 |
| 2 | batch_normalization | BatchNormalization | (256,) | 1024 | momentum=0.99, épsilon=0.001 |
| 3 | activation | Activation | (256,) | 0 | activación=relu |
| 4 | dropout | Dropout | (256,) | 0 | tasa=0.3 |
| 5 | dense_1 | Dense | (128,) | 32768 | unidades=128, activación=linear, sesgo=False, inicialización=HeNormal, L2=1e-05 |
| 6 | batch_normalization_1 | BatchNormalization | (128,) | 512 | momentum=0.99, épsilon=0.001 |
| 7 | activation_1 | Activation | (128,) | 0 | activación=relu |
| 8 | dropout_1 | Dropout | (128,) | 0 | tasa=0.3 |
| 9 | fraud_prob | Dense | (1,) | 129 | unidades=1, activación=sigmoid, sesgo=True, inicialización=GlorotUniform |

```text
Model: "mlp_regularizado"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━
┃ Layer (type)                                        ┃ Output Shape                           ┃               Para
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━
│ x (InputLayer)                                      │ (None, 383)                            │                   
├─────────────────────────────────────────────────────┼────────────────────────────────────────┼───────────────────
│ dense (Dense)                                       │ (None, 256)                            │                98,
├─────────────────────────────────────────────────────┼────────────────────────────────────────┼───────────────────
│ batch_normalization (BatchNormalization)            │ (None, 256)                            │                 1,
├─────────────────────────────────────────────────────┼────────────────────────────────────────┼───────────────────
│ activation (Activation)                             │ (None, 256)                            │                   
├─────────────────────────────────────────────────────┼────────────────────────────────────────┼───────────────────
│ dropout (Dropout)                                   │ (None, 256)                            │                   
├─────────────────────────────────────────────────────┼────────────────────────────────────────┼───────────────────
│ dense_1 (Dense)                                     │ (None, 128)                            │                32,
├─────────────────────────────────────────────────────┼────────────────────────────────────────┼───────────────────
│ batch_normalization_1 (BatchNormalization)          │ (None, 128)                            │                   
├─────────────────────────────────────────────────────┼────────────────────────────────────────┼───────────────────
│ activation_1 (Activation)                           │ (None, 128)                            │                   
├─────────────────────────────────────────────────────┼────────────────────────────────────────┼───────────────────
│ dropout_1 (Dropout)                                 │ (None, 128)                            │                   
├─────────────────────────────────────────────────────┼────────────────────────────────────────┼───────────────────
│ fraud_prob (Dense)                                  │ (None, 1)                              │                   
└─────────────────────────────────────────────────────┴────────────────────────────────────────┴───────────────────
 Total params: 132,481 (517.50 KB)
 Trainable params: 131,713 (514.50 KB)
 Non-trainable params: 768 (3.00 KB)

```

## 3. MLP con embeddings

| # | Capa | Tipo | Salida | Parámetros | Detalle |
|---|---|---|---|---|---|
| 0 | card1_idx | InputLayer | (1,) | 0 |  |
| 1 | card2_idx | InputLayer | (1,) | 0 |  |
| 2 | card3_idx | InputLayer | (1,) | 0 |  |
| 3 | card5_idx | InputLayer | (1,) | 0 |  |
| 4 | addr1_idx | InputLayer | (1,) | 0 |  |
| 5 | addr2_idx | InputLayer | (1,) | 0 |  |
| 6 | P_emaildomain_idx | InputLayer | (1,) | 0 |  |
| 7 | R_emaildomain_idx | InputLayer | (1,) | 0 |  |
| 8 | id_30_idx | InputLayer | (1,) | 0 |  |
| 9 | id_31_idx | InputLayer | (1,) | 0 |  |
| 10 | id_33_idx | InputLayer | (1,) | 0 |  |
| 11 | DeviceInfo_idx | InputLayer | (1,) | 0 |  |
| 12 | amt_price_point_idx | InputLayer | (1,) | 0 |  |
| 13 | emb_card1_idx | Embedding | (1, 32) | 30272 | vocabulario=946, dimensión=32, L2=1e-05 |
| 14 | emb_card2_idx | Embedding | (1, 32) | 14240 | vocabulario=445, dimensión=32, L2=1e-05 |
| 15 | emb_card3_idx | Embedding | (1, 11) | 319 | vocabulario=29, dimensión=11, L2=1e-05 |
| 16 | emb_card5_idx | Embedding | (1, 13) | 585 | vocabulario=45, dimensión=13, L2=1e-05 |
| 17 | emb_addr1_idx | Embedding | (1, 18) | 1296 | vocabulario=72, dimensión=18, L2=1e-05 |
| 18 | emb_addr2_idx | Embedding | (1, 5) | 35 | vocabulario=7, dimensión=5, L2=1e-05 |
| 19 | emb_P_emaildomain_idx | Embedding | (1, 15) | 810 | vocabulario=54, dimensión=15, L2=1e-05 |
| 20 | emb_R_emaildomain_idx | Embedding | (1, 12) | 420 | vocabulario=35, dimensión=12, L2=1e-05 |
| 21 | emb_id_30_idx | Embedding | (1, 16) | 928 | vocabulario=58, dimensión=16, L2=1e-05 |
| 22 | emb_id_31_idx | Embedding | (1, 17) | 1105 | vocabulario=65, dimensión=17, L2=1e-05 |
| 23 | emb_id_33_idx | Embedding | (1, 13) | 585 | vocabulario=45, dimensión=13, L2=1e-05 |
| 24 | emb_DeviceInfo_idx | Embedding | (1, 21) | 2142 | vocabulario=102, dimensión=21, L2=1e-05 |
| 25 | emb_amt_price_point_idx | Embedding | (1, 32) | 9440 | vocabulario=295, dimensión=32, L2=1e-05 |
| 26 | x | InputLayer | (383,) | 0 |  |
| 27 | flatten | Flatten | (32,) | 0 |  |
| 28 | flatten_1 | Flatten | (32,) | 0 |  |
| 29 | flatten_2 | Flatten | (11,) | 0 |  |
| 30 | flatten_3 | Flatten | (13,) | 0 |  |
| 31 | flatten_4 | Flatten | (18,) | 0 |  |
| 32 | flatten_5 | Flatten | (5,) | 0 |  |
| 33 | flatten_6 | Flatten | (15,) | 0 |  |
| 34 | flatten_7 | Flatten | (12,) | 0 |  |
| 35 | flatten_8 | Flatten | (16,) | 0 |  |
| 36 | flatten_9 | Flatten | (17,) | 0 |  |
| 37 | flatten_10 | Flatten | (13,) | 0 |  |
| 38 | flatten_11 | Flatten | (21,) | 0 |  |
| 39 | flatten_12 | Flatten | (32,) | 0 |  |
| 40 | concatenate | Concatenate | (620,) | 0 |  |
| 41 | dense | Dense | (256,) | 158720 | unidades=256, activación=linear, sesgo=False, inicialización=HeNormal, L2=1e-05 |
| 42 | batch_normalization | BatchNormalization | (256,) | 1024 | momentum=0.99, épsilon=0.001 |
| 43 | activation | Activation | (256,) | 0 | activación=relu |
| 44 | dropout | Dropout | (256,) | 0 | tasa=0.3 |
| 45 | dense_1 | Dense | (128,) | 32768 | unidades=128, activación=linear, sesgo=False, inicialización=HeNormal, L2=1e-05 |
| 46 | batch_normalization_1 | BatchNormalization | (128,) | 512 | momentum=0.99, épsilon=0.001 |
| 47 | activation_1 | Activation | (128,) | 0 | activación=relu |
| 48 | dropout_1 | Dropout | (128,) | 0 | tasa=0.3 |
| 49 | fraud_prob | Dense | (1,) | 129 | unidades=1, activación=sigmoid, sesgo=True, inicialización=GlorotUniform |

```text
Model: "mlp_embeddings"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ Layer (type)                      ┃ Output Shape                 ┃           Param # ┃ Connected to              
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ card1_idx (InputLayer)            │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ card2_idx (InputLayer)            │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ card3_idx (InputLayer)            │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ card5_idx (InputLayer)            │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ addr1_idx (InputLayer)            │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ addr2_idx (InputLayer)            │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ P_emaildomain_idx (InputLayer)    │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ R_emaildomain_idx (InputLayer)    │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ id_30_idx (InputLayer)            │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ id_31_idx (InputLayer)            │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ id_33_idx (InputLayer)            │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ DeviceInfo_idx (InputLayer)       │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ amt_price_point_idx (InputLayer)  │ (None, 1)                    │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_card1_idx (Embedding)         │ (None, 1, 32)                │            30,272 │ card1_idx[0][0]           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_card2_idx (Embedding)         │ (None, 1, 32)                │            14,240 │ card2_idx[0][0]           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_card3_idx (Embedding)         │ (None, 1, 11)                │               319 │ card3_idx[0][0]           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_card5_idx (Embedding)         │ (None, 1, 13)                │               585 │ card5_idx[0][0]           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_addr1_idx (Embedding)         │ (None, 1, 18)                │             1,296 │ addr1_idx[0][0]           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_addr2_idx (Embedding)         │ (None, 1, 5)                 │                35 │ addr2_idx[0][0]           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_P_emaildomain_idx (Embedding) │ (None, 1, 15)                │               810 │ P_emaildomain_idx[0][0]   
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_R_emaildomain_idx (Embedding) │ (None, 1, 12)                │               420 │ R_emaildomain_idx[0][0]   
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_id_30_idx (Embedding)         │ (None, 1, 16)                │               928 │ id_30_idx[0][0]           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_id_31_idx (Embedding)         │ (None, 1, 17)                │             1,105 │ id_31_idx[0][0]           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_id_33_idx (Embedding)         │ (None, 1, 13)                │               585 │ id_33_idx[0][0]           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_DeviceInfo_idx (Embedding)    │ (None, 1, 21)                │             2,142 │ DeviceInfo_idx[0][0]      
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ emb_amt_price_point_idx           │ (None, 1, 32)                │             9,440 │ amt_price_point_idx[0][0] 
│ (Embedding)                       │                              │                   │                           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ x (InputLayer)                    │ (None, 383)                  │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten (Flatten)                 │ (None, 32)                   │                 0 │ emb_card1_idx[0][0]       
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten_1 (Flatten)               │ (None, 32)                   │                 0 │ emb_card2_idx[0][0]       
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten_2 (Flatten)               │ (None, 11)                   │                 0 │ emb_card3_idx[0][0]       
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten_3 (Flatten)               │ (None, 13)                   │                 0 │ emb_card5_idx[0][0]       
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten_4 (Flatten)               │ (None, 18)                   │                 0 │ emb_addr1_idx[0][0]       
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten_5 (Flatten)               │ (None, 5)                    │                 0 │ emb_addr2_idx[0][0]       
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten_6 (Flatten)               │ (None, 15)                   │                 0 │ emb_P_emaildomain_idx[0][0
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten_7 (Flatten)               │ (None, 12)                   │                 0 │ emb_R_emaildomain_idx[0][0
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten_8 (Flatten)               │ (None, 16)                   │                 0 │ emb_id_30_idx[0][0]       
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten_9 (Flatten)               │ (None, 17)                   │                 0 │ emb_id_31_idx[0][0]       
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten_10 (Flatten)              │ (None, 13)                   │                 0 │ emb_id_33_idx[0][0]       
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten_11 (Flatten)              │ (None, 21)                   │                 0 │ emb_DeviceInfo_idx[0][0]  
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ flatten_12 (Flatten)              │ (None, 32)                   │                 0 │ emb_amt_price_point_idx[0]
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ concatenate (Concatenate)         │ (None, 620)                  │                 0 │ x[0][0], flatten[0][0],   
│                                   │                              │                   │ flatten_1[0][0],          
│                                   │                              │                   │ flatten_2[0][0],          
│                                   │                              │                   │ flatten_3[0][0],          
│                                   │                              │                   │ flatten_4[0][0],          
│                                   │                              │                   │ flatten_5[0][0],          
│                                   │                              │                   │ flatten_6[0][0],          
│                                   │                              │                   │ flatten_7[0][0],          
│                                   │                              │                   │ flatten_8[0][0],          
│                                   │                              │                   │ flatten_9[0][0],          
│                                   │                              │                   │ flatten_10[0][0],         
│                                   │                              │                   │ flatten_11[0][0],         
│                                   │                              │                   │ flatten_12[0][0]          
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dense (Dense)                     │ (None, 256)                  │           158,720 │ concatenate[0][0]         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ batch_normalization               │ (None, 256)                  │             1,024 │ dense[0][0]               
│ (BatchNormalization)              │                              │                   │                           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ activation (Activation)           │ (None, 256)                  │                 0 │ batch_normalization[0][0] 
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dropout (Dropout)                 │ (None, 256)                  │                 0 │ activation[0][0]          
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dense_1 (Dense)                   │ (None, 128)                  │            32,768 │ dropout[0][0]             
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ batch_normalization_1             │ (None, 128)                  │               512 │ dense_1[0][0]             
│ (BatchNormalization)              │                              │                   │                           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ activation_1 (Activation)         │ (None, 128)                  │                 0 │ batch_normalization_1[0][0
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dropout_1 (Dropout)               │ (None, 128)                  │                 0 │ activation_1[0][0]        
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ fraud_prob (Dense)                │ (None, 1)                    │               129 │ dropout_1[0][0]           
└───────────────────────────────────┴──────────────────────────────┴───────────────────┴───────────────────────────
 Total params: 255,330 (997.38 KB)
 Trainable params: 254,562 (994.38 KB)
 Non-trainable params: 768 (3.00 KB)

```

## 4. Residual MLP

| # | Capa | Tipo | Salida | Parámetros | Detalle |
|---|---|---|---|---|---|
| 0 | x | InputLayer | (383,) | 0 |  |
| 1 | dense | Dense | (256,) | 98048 | unidades=256, activación=linear, sesgo=False, inicialización=HeNormal, L2=1e-05 |
| 2 | batch_normalization | BatchNormalization | (256,) | 1024 | momentum=0.99, épsilon=0.001 |
| 3 | activation | Activation | (256,) | 0 | activación=relu |
| 4 | dense_1 | Dense | (256,) | 65536 | unidades=256, activación=linear, sesgo=False, inicialización=HeNormal, L2=1e-05 |
| 5 | batch_normalization_1 | BatchNormalization | (256,) | 1024 | momentum=0.99, épsilon=0.001 |
| 6 | activation_1 | Activation | (256,) | 0 | activación=relu |
| 7 | dropout | Dropout | (256,) | 0 | tasa=0.2 |
| 8 | dense_2 | Dense | (256,) | 65536 | unidades=256, activación=linear, sesgo=False, inicialización=HeNormal, L2=1e-05 |
| 9 | batch_normalization_2 | BatchNormalization | (256,) | 1024 | momentum=0.99, épsilon=0.001 |
| 10 | add | Add | (256,) | 0 |  |
| 11 | activation_2 | Activation | (256,) | 0 | activación=relu |
| 12 | dense_3 | Dense | (256,) | 65536 | unidades=256, activación=linear, sesgo=False, inicialización=HeNormal, L2=1e-05 |
| 13 | batch_normalization_3 | BatchNormalization | (256,) | 1024 | momentum=0.99, épsilon=0.001 |
| 14 | activation_3 | Activation | (256,) | 0 | activación=relu |
| 15 | dropout_1 | Dropout | (256,) | 0 | tasa=0.2 |
| 16 | dense_4 | Dense | (256,) | 65536 | unidades=256, activación=linear, sesgo=False, inicialización=HeNormal, L2=1e-05 |
| 17 | batch_normalization_4 | BatchNormalization | (256,) | 1024 | momentum=0.99, épsilon=0.001 |
| 18 | add_1 | Add | (256,) | 0 |  |
| 19 | activation_4 | Activation | (256,) | 0 | activación=relu |
| 20 | dense_5 | Dense | (256,) | 65536 | unidades=256, activación=linear, sesgo=False, inicialización=HeNormal, L2=1e-05 |
| 21 | batch_normalization_5 | BatchNormalization | (256,) | 1024 | momentum=0.99, épsilon=0.001 |
| 22 | activation_5 | Activation | (256,) | 0 | activación=relu |
| 23 | dropout_2 | Dropout | (256,) | 0 | tasa=0.2 |
| 24 | dense_6 | Dense | (256,) | 65536 | unidades=256, activación=linear, sesgo=False, inicialización=HeNormal, L2=1e-05 |
| 25 | batch_normalization_6 | BatchNormalization | (256,) | 1024 | momentum=0.99, épsilon=0.001 |
| 26 | add_2 | Add | (256,) | 0 |  |
| 27 | activation_6 | Activation | (256,) | 0 | activación=relu |
| 28 | fraud_prob | Dense | (1,) | 257 | unidades=1, activación=sigmoid, sesgo=True, inicialización=GlorotUniform |

```text
Model: "residual_mlp"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ Layer (type)                      ┃ Output Shape                 ┃           Param # ┃ Connected to              
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ x (InputLayer)                    │ (None, 383)                  │                 0 │ -                         
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dense (Dense)                     │ (None, 256)                  │            98,048 │ x[0][0]                   
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ batch_normalization               │ (None, 256)                  │             1,024 │ dense[0][0]               
│ (BatchNormalization)              │                              │                   │                           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ activation (Activation)           │ (None, 256)                  │                 0 │ batch_normalization[0][0] 
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dense_1 (Dense)                   │ (None, 256)                  │            65,536 │ activation[0][0]          
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ batch_normalization_1             │ (None, 256)                  │             1,024 │ dense_1[0][0]             
│ (BatchNormalization)              │                              │                   │                           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ activation_1 (Activation)         │ (None, 256)                  │                 0 │ batch_normalization_1[0][0
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dropout (Dropout)                 │ (None, 256)                  │                 0 │ activation_1[0][0]        
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dense_2 (Dense)                   │ (None, 256)                  │            65,536 │ dropout[0][0]             
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ batch_normalization_2             │ (None, 256)                  │             1,024 │ dense_2[0][0]             
│ (BatchNormalization)              │                              │                   │                           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ add (Add)                         │ (None, 256)                  │                 0 │ activation[0][0],         
│                                   │                              │                   │ batch_normalization_2[0][0
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ activation_2 (Activation)         │ (None, 256)                  │                 0 │ add[0][0]                 
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dense_3 (Dense)                   │ (None, 256)                  │            65,536 │ activation_2[0][0]        
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ batch_normalization_3             │ (None, 256)                  │             1,024 │ dense_3[0][0]             
│ (BatchNormalization)              │                              │                   │                           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ activation_3 (Activation)         │ (None, 256)                  │                 0 │ batch_normalization_3[0][0
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dropout_1 (Dropout)               │ (None, 256)                  │                 0 │ activation_3[0][0]        
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dense_4 (Dense)                   │ (None, 256)                  │            65,536 │ dropout_1[0][0]           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ batch_normalization_4             │ (None, 256)                  │             1,024 │ dense_4[0][0]             
│ (BatchNormalization)              │                              │                   │                           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ add_1 (Add)                       │ (None, 256)                  │                 0 │ activation_2[0][0],       
│                                   │                              │                   │ batch_normalization_4[0][0
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ activation_4 (Activation)         │ (None, 256)                  │                 0 │ add_1[0][0]               
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dense_5 (Dense)                   │ (None, 256)                  │            65,536 │ activation_4[0][0]        
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ batch_normalization_5             │ (None, 256)                  │             1,024 │ dense_5[0][0]             
│ (BatchNormalization)              │                              │                   │                           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ activation_5 (Activation)         │ (None, 256)                  │                 0 │ batch_normalization_5[0][0
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dropout_2 (Dropout)               │ (None, 256)                  │                 0 │ activation_5[0][0]        
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ dense_6 (Dense)                   │ (None, 256)                  │            65,536 │ dropout_2[0][0]           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ batch_normalization_6             │ (None, 256)                  │             1,024 │ dense_6[0][0]             
│ (BatchNormalization)              │                              │                   │                           
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ add_2 (Add)                       │ (None, 256)                  │                 0 │ activation_4[0][0],       
│                                   │                              │                   │ batch_normalization_6[0][0
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ activation_6 (Activation)         │ (None, 256)                  │                 0 │ add_2[0][0]               
├───────────────────────────────────┼──────────────────────────────┼───────────────────┼───────────────────────────
│ fraud_prob (Dense)                │ (None, 1)                    │               257 │ activation_6[0][0]        
└───────────────────────────────────┴──────────────────────────────┴───────────────────┴───────────────────────────
 Total params: 498,689 (1.90 MB)
 Trainable params: 495,105 (1.89 MB)
 Non-trainable params: 3,584 (14.00 KB)

```