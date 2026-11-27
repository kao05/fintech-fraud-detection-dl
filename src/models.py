"""Arquitecturas de red (Etapa 3).

Cada función CONSTRUYE el modelo y no lo compila: la pérdida depende de la estrategia
de costo y la elige src/train.py. Todas terminan en la capa 'fraud_prob' (sigmoid).
"""
from __future__ import annotations

import keras
from keras import layers, regularizers


def _l2(v):
    return regularizers.l2(v) if v else None


def _block(x, units, dropout=0.0, l2=0.0, batch_norm=True):
    """Dense -> (BN) -> ReLU -> (Dropout)."""
    x = layers.Dense(units, use_bias=not batch_norm, kernel_initializer="he_normal",
                     kernel_regularizer=_l2(l2))(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x


def _head(x):
    return layers.Dense(1, activation="sigmoid", name="fraud_prob")(x)


def embedding_dim(vocab: int, cap: int = 32, factor: float = 1.6, power: float = 0.56) -> int:
    return int(min(cap, max(1, round(factor * vocab ** power))))


# 1. MLP convencional (ablación): Dense + ReLU, inicialización He, sin regularización
def build_mlp(n_features, hidden=(256, 128), name="mlp"):
    inp = keras.Input(shape=(n_features,), name="x")
    x = inp
    for u in hidden:
        x = layers.Dense(u, activation="relu", kernel_initializer="he_normal")(x)
    return keras.Model(inp, _head(x), name=name)


# 2. MLP regularizada: L2 + Dropout + Batch Normalization (el early stopping está en train.py)
def build_mlp_regularizado(n_features, hidden=(256, 128), dropout=0.3, l2=1e-5,
                           batch_norm=True, name="mlp_regularizado"):
    inp = keras.Input(shape=(n_features,), name="x")
    x = inp
    for u in hidden:
        x = _block(x, u, dropout, l2, batch_norm)
    return keras.Model(inp, _head(x), name=name)


# 3. MLP con embeddings: un Embedding por columna de idx, concatenado con X
def build_mlp_embeddings(n_features, vocab_sizes, hidden=(256, 128), dropout=0.3, l2=1e-5,
                         batch_norm=True, emb=None, name="mlp_embeddings"):
    emb = {"cap": 32, "factor": 1.6, "power": 0.56, **(emb or {})}
    x_in = keras.Input(shape=(n_features,), name="x")
    inputs, parts = {"x": x_in}, [x_in]
    for col, vocab in vocab_sizes.items():
        i = keras.Input(shape=(1,), dtype="int32", name=col)
        inputs[col] = i
        e = layers.Embedding(vocab, embedding_dim(vocab, **emb),
                             embeddings_regularizer=_l2(l2), name=f"emb_{col}")(i)
        parts.append(layers.Flatten()(e))
    x = layers.Concatenate()(parts)
    for u in hidden:
        x = _block(x, u, dropout, l2, batch_norm)
    return keras.Model(inputs, _head(x), name=name)


# 4. Residual MLP: proyección inicial + bloques con conexión residual
def build_residual_mlp(n_features, width=256, n_blocks=3, dropout=0.2, l2=1e-5, name="residual_mlp"):
    inp = keras.Input(shape=(n_features,), name="x")
    x = _block(inp, width, 0.0, l2, True)                      # proyección inicial
    for _ in range(n_blocks):
        h = _block(x, width, dropout, l2, True)                # Dense -> BN -> ReLU -> Dropout
        h = layers.Dense(width, use_bias=False, kernel_initializer="he_normal",
                         kernel_regularizer=_l2(l2))(h)
        h = layers.BatchNormalization()(h)
        x = layers.Activation("relu")(layers.Add()([x, h]))    # conexión residual
    return keras.Model(inp, _head(x), name=name)


KINDS = {
    "mlp": build_mlp,
    "mlp_regularizado": build_mlp_regularizado,
    "mlp_embeddings": build_mlp_embeddings,
    "residual_mlp": build_residual_mlp,
}


def build_model(kind: str, cfg: dict, n_features: int, vocab_sizes: dict | None = None, **overrides):
    """Construye una arquitectura con los valores de la config; `overrides` pisa hiperparámetros."""
    params = {**cfg["nn"]["architectures"][kind], **overrides}
    if "hidden" in params:
        params["hidden"] = tuple(params["hidden"])
    if kind == "mlp_embeddings":
        params.setdefault("emb", dict(cfg["nn"]["embedding_dim"]))
        return build_mlp_embeddings(n_features, vocab_sizes, **params)
    return KINDS[kind](n_features, **params)