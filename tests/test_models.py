import numpy as np
import pandas as pd
import pytest

pytest.importorskip("keras")
from src.models import (build_mlp, build_mlp_embeddings, build_mlp_regularizado,
                        build_residual_mlp, embedding_dim)
from src.train import initial_output_bias, make_inputs

N_F = 12
VOCAB = {"a_idx": 10, "b_idx": 300}


def _data(n=64):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, N_F)).astype("float32")
    idx = pd.DataFrame({k: rng.integers(0, v, size=n) for k, v in VOCAB.items()})
    return X, idx


@pytest.mark.parametrize("builder", [build_mlp, build_mlp_regularizado, build_residual_mlp])
def test_mlp_tabulares_salida_sigmoid(builder):
    X, _ = _data()
    p = builder(N_F).predict(make_inputs(X), verbose=0)
    assert p.shape == (64, 1) and ((p > 0) & (p < 1)).all()


def test_mlp_embeddings_acepta_diccionario():
    X, idx = _data()
    model = build_mlp_embeddings(N_F, VOCAB)
    p = model.predict(make_inputs(X, idx, "mlp_embeddings", list(VOCAB)), verbose=0)
    assert p.shape == (64, 1)
    assert {inp.name for inp in model.inputs} == {"x", *VOCAB}


def test_dimension_de_embedding():
    assert embedding_dim(946) == 32            # tope
    assert 1 <= embedding_dim(7) <= 6


def test_sesgo_inicial():
    y = np.array([0] * 90 + [1] * 10)
    assert initial_output_bias(y, "class_weight") == 0.0
    assert initial_output_bias(y, "focal") == pytest.approx(np.log(10 / 90))