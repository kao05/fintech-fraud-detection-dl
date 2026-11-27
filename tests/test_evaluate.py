import numpy as np
import pytest

from src.evaluate import (
    SCHEMA, best_f1_threshold, confusion, evaluate_scores, expected_cost,
    log_result, min_cost_threshold, pr_auc, precision_recall_f1,
    recall_at_precision, roc_auc,
)

Y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
P = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.1, 0.05, 0.05, 0.0])
CFG = {"cost": {"c_FN": 10, "c_FP": 1, "target_precision": 0.5}}


def test_clasificador_perfecto():
    y = np.array([0, 0, 1, 1, 0, 1])
    assert pr_auc(y, y.astype(float)) == pytest.approx(1.0)
    assert roc_auc(y, y.astype(float)) == pytest.approx(1.0)


def test_clasificador_aleatorio_cerca_de_la_tasa():
    rng = np.random.default_rng(0)
    y = (rng.random(100_000) < 0.035).astype(int)
    assert abs(pr_auc(y, rng.random(100_000)) - y.mean()) < 0.005


def test_matriz_y_prf1_a_mano():
    assert confusion(Y, P, 0.5) == {"tn": 6, "fp": 1, "fn": 1, "tp": 2}
    prec, rec, f1 = precision_recall_f1(Y, P, 0.5)
    assert prec == pytest.approx(2 / 3) and rec == pytest.approx(2 / 3) and f1 == pytest.approx(2 / 3)


def test_costo_esperado_a_mano():
    assert expected_cost(Y, P, 0.5, 10, 1) == 11.0


def test_costo_minimo():
    thr, cost = min_cost_threshold(Y, P, 10, 1)
    assert thr == pytest.approx(0.3) and cost == 1.0


def test_costo_minimo_sin_mejora_devuelve_no_marcar_nada():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.1, 0.9, 0.8, 0.7])      # el fraude puntúa más bajo: marcar solo cuesta
    thr, cost = min_cost_threshold(y, p, c_FN=1, c_FP=10)
    assert np.isinf(thr) and cost == 1.0


def test_recall_a_precision_fija():
    assert recall_at_precision(Y, P, 0.5) == pytest.approx(1.0)
    assert recall_at_precision(Y, P, 0.9) == pytest.approx(2 / 3)
    assert recall_at_precision(np.array([1, 0]), np.array([0.1, 0.9]), 0.9) == 0.0


def test_mejor_f1():
    thr, f1 = best_f1_threshold(Y, P)
    assert f1 >= 0.85   # el umbral 0.3 da P=0.75, R=1 -> F1=0.857


def test_evaluate_scores_claves():
    m = evaluate_scores(Y, P, CFG)
    for k in ["pr_auc", "roc_auc", "recall_at_p", "f1_max", "min_cost", "min_cost_rel", "pr_floor"]:
        assert k in m
    assert m["pr_floor"] == pytest.approx(0.3)


def test_log_result(tmp_path):
    f = tmp_path / "results.csv"
    a = log_result({"stage": "baseline", "model": "logreg", "pr_auc": 0.3, "hyperparams": {"C": 1}}, f)
    b = log_result({"stage": "baseline", "model": "xgb", "pr_auc": 0.5}, f)
    assert (a, b) == ("exp_0001", "exp_0002")
    assert list(__import__("pandas").read_csv(f).columns) == SCHEMA
    with pytest.raises(ValueError):
        log_result({"stage": "baseline", "model": "x"}, f)