"""Pruebas anti-fuga (Acción 11). Datos sintéticos pequeños: corren en segundos.

Desde la raíz del proyecto:  pytest tests/test_no_leakage.py -v
"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.data import load_config
from src.preprocessing import Preprocessor, prepare_frame
from src.split import temporal_split
from src.train import compute_class_weights

N = 6000
PRICES = np.array([9.99, 25.0, 57.95, 100.0, 150.0])
FORBIDDEN = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]


# ------------------------------------------------------------------ datos sintéticos
def _nan_where(values, mask):
    return np.where(mask, np.nan, values).astype("float32")


def make_df(n: int = N, seed: int = 7) -> pd.DataFrame:
    """Misma estructura que el dataset real: deriva temporal, nulos estructurales, categóricas de
    alta y baja cardinalidad, precios recurrentes y empates de TransactionDT en el corte."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(86_400, 86_400 * 100, size=n))
    cut = int(n * 0.70)
    t[cut - 10: cut + 10] = t[cut]                          # empate justo en el corte train/dev (A8)
    fraud = (rng.random(n) < 0.03 + 0.03 * np.linspace(0, 1, n)).astype("int8")   # deriva
    has_id = (rng.random(n) < 0.25).astype("float32")
    no_id = has_id == 0
    m_v = rng.random(n) < 0.20                              # V1 y V2 comparten patrón de nulos
    amt = np.where(rng.random(n) < 0.8, rng.choice(PRICES, n), np.round(rng.lognormal(4, 1, n), 2))
    v1 = rng.normal(size=n)
    devices = np.array([f"dev_{i:02d}" for i in range(40)], dtype=object)
    domains = np.array([f"mail{i}.com" for i in range(12)], dtype=object)
    return pd.DataFrame({
        "TransactionID": np.arange(10_000, 10_000 + n),
        "isFraud": fraud,
        "TransactionDT": t.astype("float32"),
        "TransactionAmt": amt.astype("float32"),
        "has_identity": has_id,
        "C1": (rng.pareto(1.5, n) * 3).astype("float32"),                    # cola larga
        "C2": (rng.normal(size=n) + 1.5 * fraud).astype("float32"),
        "D1": _nan_where(rng.exponential(10, n), rng.random(n) < 0.30),
        "D2": rng.normal(size=n).astype("float32"),                          # SIN nulos en train
        "V1": _nan_where(v1, m_v),
        "V2": _nan_where(2 * v1 + rng.normal(scale=0.05, size=n), m_v),
        "id_01": _nan_where(rng.normal(size=n), no_id),
        "id_02": _nan_where(rng.normal(size=n), no_id),
        "ProductCD": rng.choice(["W", "C", "R", "H"], n, p=[.6, .2, .15, .05]).astype(object),
        "card4": np.where(rng.random(n) < 0.02, None,
                          rng.choice(["visa", "mastercard", "discover"], n)).astype(object),
        "DeviceInfo": np.where(no_id, None, rng.choice(devices, n)).astype(object),
        "P_emaildomain": np.where(rng.random(n) < 0.10, None, rng.choice(domains, n)).astype(object),
        "card1": _nan_where(rng.integers(1000, 1060, n), rng.random(n) < 0.01),
    })


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config())
    c["preprocessing"].update(min_indicator_rows=5, min_category_freq=5,
                              amt_price_point_min_freq=5, onehot_max_cardinality=5)
    return c


@pytest.fixture(scope="module")
def df():
    return make_df()


@pytest.fixture(scope="module")
def splits(df, cfg):
    return temporal_split(df, cfg)


@pytest.fixture(scope="module")
def fitted(splits, cfg):
    train, dev, test = splits
    prep = Preprocessor(cfg)
    out_train = prep.fit_transform(train)           # único ajuste: solo train
    return prep, out_train, prep.transform(dev), prep.transform(test)


def state_snapshot(prep) -> dict:
    """Todo lo que el Preprocessor 'aprendió' de train."""
    return copy.deepcopy({
        "fill": prep.imputer.fill_, "mean": prep.numtf.mean_, "std": prep.numtf.std_,
        "log": prep.numtf.log_cols_, "vocab": prep.encoder.vocab_, "freq": prep.encoder.freq_,
        "amt_freq": prep.encoder.amt_freq_, "price": prep.encoder.price_vocab_,
        "groups": prep.nulls.groups_, "keep": prep.rf.keep_cols_, "features": prep.feature_names_,
    })


# ------------------------------------------------------------------ 1. split temporal
def test_split_is_temporal_and_disjoint(splits, df):
    train, dev, test = splits
    assert train["TransactionDT"].max() < dev["TransactionDT"].min()
    assert dev["TransactionDT"].max() < test["TransactionDT"].min()
    ids = [set(s["TransactionID"]) for s in splits]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(train) + len(dev) + len(test) == len(df)           # no se pierde ni se duplica nada


def test_ties_in_time_are_not_split(splits, df, cfg):
    tie_t = df["TransactionDT"].iloc[int(len(df) * cfg["split"]["train"])]
    counts = [int((s["TransactionDT"] == tie_t).sum()) for s in splits]
    assert sum(counts) == int((df["TransactionDT"] == tie_t).sum())
    assert sum(c > 0 for c in counts) == 1, "Un empate de tiempo quedó repartido entre splits"


def test_split_fractions(splits, df, cfg):
    train, dev, test = splits
    assert abs(len(train) / len(df) - cfg["split"]["train"]) < 0.01
    assert abs(len(dev) / len(df) - cfg["split"]["dev"]) < 0.01
    assert abs(len(test) / len(df) - cfg["split"]["test"]) < 0.01


def test_split_does_not_depend_on_row_order(splits, df, cfg):
    shuffled = temporal_split(df.sample(frac=1, random_state=0), cfg)
    for a, b in zip(splits, shuffled):
        assert set(a["TransactionID"]) == set(b["TransactionID"])


# ------------------------------------------------------------------ 2. salidas del pipeline
def test_forbidden_columns_not_in_features_and_ids_disjoint(fitted):
    prep, o_tr, o_dev, o_te = fitted
    for o in (o_tr, o_dev, o_te):
        assert not set(FORBIDDEN) & set(o["X"].columns)
        assert not set(FORBIDDEN) & set(o["idx"].columns)
        assert list(o["meta"].columns) == ["TransactionID", "TransactionDT", "TransactionAmt"]
    ids = [set(o["meta"]["TransactionID"]) for o in (o_tr, o_dev, o_te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert (prep.feature_names_ and "has_identity" in prep.feature_names_)


def test_same_columns_no_nan_no_inf_and_no_resampling(fitted, splits):
    _, o_tr, o_dev, o_te = fitted
    assert list(o_tr["X"].columns) == list(o_dev["X"].columns) == list(o_te["X"].columns)
    assert list(o_tr["idx"].columns) == list(o_dev["idx"].columns) == list(o_te["idx"].columns)
    for o, s in zip((o_tr, o_dev, o_te), splits):
        assert np.isfinite(o["X"].to_numpy(dtype="float64")).all()
        assert not o["idx"].isna().any().any()
        assert len(o["X"]) == len(o["idx"]) == len(o["y"]) == len(s)       # nada de remuestreo
        assert set(np.unique(o["y"])) <= {0, 1}


# ------------------------------------------------------------------ 3. todo se ajusta solo con train
def test_medians_vocab_and_null_patterns_come_from_train(fitted, splits):
    prep, *_ = fitted
    train, dev, _ = splits
    # mediana de imputación = mediana de train calculada de forma independiente
    assert prep.imputer.fill_["D1"] == pytest.approx(float(train["D1"].median()), rel=1e-5)
    # vocabulario de una categórica de alta cardinalidad ⊆ categorías vistas en train (+ 'missing')
    seen = set(train["DeviceInfo"].dropna().unique()) | {"missing"}
    assert set(prep.encoder.vocab_["DeviceInfo"]) <= seen
    # todo grupo de indicadores se formó con columnas que tienen nulos en train
    for cols in prep.nulls.groups_.values():
        assert all(train[c].isna().any() for c in cols)


def test_dev_is_scaled_with_train_statistics(fitted, splits, cfg):
    prep, _, o_dev, _ = fitted
    train, dev, _ = splits
    col = "C2"
    assert col in prep.numtf.scale_cols_ and col not in prep.numtf.log_cols_
    assert col in o_dev["X"].columns, "C2 no debería ser eliminada por redundancia"
    v_tr = train[col].to_numpy(dtype="float64")
    mu, sd = prep.numtf.mean_[col], prep.numtf.std_[col]
    assert mu == pytest.approx(v_tr.mean(), rel=1e-6)              # estadísticos = los de train
    assert sd == pytest.approx(v_tr.std(), rel=1e-6)
    expected = (dev[col].to_numpy(dtype="float64") - mu) / sd      # dev con la media/desv. de TRAIN
    clip = cfg["preprocessing"].get("clip_z")
    if clip is not None:
        expected = np.clip(expected, -clip, clip)
    np.testing.assert_allclose(o_dev["X"][col].to_numpy(dtype="float64"), expected, atol=1e-5)


def test_transform_does_not_change_fitted_state(fitted, splits):
    prep, o_tr, _, _ = fitted
    train, dev, test = splits
    before = state_snapshot(prep)
    for d in (dev, test):                                          # dev/test extremos y con categorías nuevas
        ext = d.copy()
        ext["C1"] = ext["C1"] * 1e6
        ext["D1"] = np.float32(np.nan)
        ext["DeviceInfo"] = "NEW_DEVICE"
        ext["card1"] = np.float32(99_999)
        out = prep.transform(ext)
        assert np.isfinite(out["X"].to_numpy(dtype="float64")).all()
    assert state_snapshot(prep) == before, "transform modificó el estado ajustado con train"
    pd.testing.assert_frame_equal(prep.transform(train)["X"], o_tr["X"])   # train no cambia por dev/test


# ------------------------------------------------------------------ 4. categorías y nulos nuevos
def test_unseen_categories_go_to_unknown(fitted, splits, cfg):
    prep, *_ = fitted
    _, dev, _ = splits
    d = dev.copy()
    d.loc[d.index[:50], "DeviceInfo"] = "NEW_DEVICE"
    out = prep.transform(d)
    assert (out["idx"].loc[:49, "DeviceInfo_idx"] == 0).all()      # índice 0 = desconocido
    assert "NEW_DEVICE" not in prep.encoder.vocab_["DeviceInfo"]   # el vocabulario no se amplió
    X, _, meta = prepare_frame(d, cfg)
    E, _ = prep.encoder.transform(prep.imputer.transform(X), meta[cfg["columns"]["amount"]])
    assert (E.loc[:49, "DeviceInfo_freq"] == 0).all()              # frecuencia de lo nuevo = 0


def test_columns_without_train_nulls_get_no_indicator(fitted, splits, cfg):
    prep, *_ = fitted
    _, dev, _ = splits
    assert all("D2" not in cols for cols in prep.nulls.groups_.values())
    d = dev.copy()
    d.loc[d.index[:100], "D2"] = np.float32(np.nan)                # nulos que train nunca vio
    out = prep.transform(d)
    assert not out["X"].isna().any().any()                         # igual se imputan (mediana de train)
    X, _, _ = prepare_frame(d, cfg)
    assert "D2" in prep.nulls.uncovered_null_columns(X)            # y queda registrado como sin cobertura


# ------------------------------------------------------------------ 5. orden de uso y pesos
def test_transform_requires_fit(splits, cfg):
    _, dev, _ = splits
    with pytest.raises(AssertionError):
        Preprocessor(cfg).transform(dev)


def test_class_weights_use_train_only(splits, cfg):
    train, *_ = splits
    cw = compute_class_weights(train["isFraud"], cfg["imbalance"]["class_weight_mode"])
    n1 = int(train["isFraud"].sum())
    n0 = len(train) - n1
    assert (cw["n0"], cw["n1"]) == (n0, n1)
    assert cw["scale_pos_weight"] == pytest.approx(n0 / n1)