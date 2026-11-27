"""Métricas y registro de experimentos (Etapa 3, primera versión).

Todos los umbrales usan la regla  p >= thr.  En esta etapa se eligen y se
reportan en dev; la evaluación con umbral fijo sobre test es de la Etapa 4.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = [
    "exp_id", "timestamp", "stage", "model", "variant", "cost_strategy", "seed",
    "hyperparams", "n_features", "best_iter", "train_time_s",
    "pr_auc_train", "pr_auc", "roc_auc", "recall_at_p", "f1_max",
    "min_cost", "min_cost_rel", "notes",
]
REQUIRED = ["stage", "model", "pr_auc"]


def _arr(x) -> np.ndarray:
    return np.asarray(x).ravel()


# ---------------------------------------------------------------- métricas
def pr_auc(y, p) -> float:
    return float(average_precision_score(_arr(y), _arr(p)))


def roc_auc(y, p) -> float:
    return float(roc_auc_score(_arr(y), _arr(p)))


def confusion(y, p, thr: float) -> dict:
    y = _arr(y).astype(int)
    pred = (_arr(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def precision_recall_f1(y, p, thr: float):
    c = confusion(y, p, thr)
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def recall_at_precision(y, p, target: float, return_threshold: bool = False):
    """Mayor recall entre los umbrales con precisión >= target (0 si ninguno)."""
    prec, rec, thr = precision_recall_curve(_arr(y), _arr(p))
    mask = prec[:-1] >= target          # el último punto (prec=1, rec=0) no tiene umbral
    if not mask.any():
        return (0.0, np.inf) if return_threshold else 0.0
    i = int(np.argmax(np.where(mask, rec[:-1], -1.0)))
    r = float(rec[:-1][i])
    return (r, float(thr[i])) if return_threshold else r


def best_f1_threshold(y, p):
    """(umbral, F1) que maximiza F1."""
    prec, rec, thr = precision_recall_curve(_arr(y), _arr(p))
    prec, rec = prec[:-1], rec[:-1]
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def expected_cost(y, p, thr: float, c_FN: float, c_FP: float) -> float:
    c = confusion(y, p, thr)
    return float(c_FN * c["fn"] + c_FP * c["fp"])


def min_cost_threshold(y, p, c_FN: float, c_FP: float):
    """(umbral, costo) del mínimo costo esperado.

    Si ni marcando nada se mejora el costo, devuelve (inf, c_FN * fraudes).
    """
    y = _arr(y).astype(int)
    p = _arr(p)
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]   # último índice de cada score único
    tp, fp, thr = tp[last], fp[last], ps[last]
    cost = c_FN * (ys.sum() - tp) + c_FP * fp
    i = int(np.argmin(cost))
    base = float(c_FN * ys.sum())
    if base <= cost[i]:
        return float("inf"), base
    return float(thr[i]), float(cost[i])


def evaluate_scores(y, p, cfg: dict) -> dict:
    """Todas las métricas de un modelo (umbrales elegidos con los mismos datos que se pasan)."""
    y, p = _arr(y).astype(int), _arr(p)
    c_FN, c_FP = cfg["cost"]["c_FN"], cfg["cost"]["c_FP"]
    target = cfg["cost"]["target_precision"]

    rec_p, thr_rec = recall_at_precision(y, p, target, return_threshold=True)
    thr_f1, f1 = best_f1_threshold(y, p)
    prec_f1, rec_f1, _ = precision_recall_f1(y, p, thr_f1)
    thr_cost, min_cost = min_cost_threshold(y, p, c_FN, c_FP)
    base_cost = float(c_FN * y.sum())

    return {
        "n": int(len(y)), "n_fraud": int(y.sum()), "pr_floor": float(y.mean()),
        "pr_auc": pr_auc(y, p), "roc_auc": roc_auc(y, p),
        "recall_at_p": rec_p, "thr_recall_at_p": thr_rec,
        "f1_max": f1, "thr_f1": thr_f1, "precision_at_f1": prec_f1, "recall_at_f1": rec_f1,
        "min_cost": min_cost, "thr_cost": thr_cost,
        "base_cost": base_cost, "min_cost_rel": min_cost / base_cost if base_cost > 0 else np.nan,
    }


# ---------------------------------------------------------------- registro
def _resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def log_result(row: dict, path="reports/results.csv") -> str:
    """Agrega una fila a results.csv. Devuelve el exp_id.

    Rellena exp_id y timestamp; serializa hyperparams a JSON; ignora claves fuera
    del esquema; exige stage, model y pr_auc (no deja filas incompletas).
    """
    missing = [k for k in REQUIRED if row.get(k) in (None, "")]
    if missing:
        raise ValueError(f"Faltan campos obligatorios: {missing}")

    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_prev = len(pd.read_csv(path)) if path.exists() else 0

    out = {k: row.get(k, "") for k in SCHEMA}
    out["exp_id"] = f"exp_{n_prev + 1:04d}"
    out["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if not isinstance(out["hyperparams"], str):
        out["hyperparams"] = json.dumps(out["hyperparams"] or {}, sort_keys=True, default=str)

    pd.DataFrame([out], columns=SCHEMA).to_csv(path, mode="a", header=not path.exists(), index=False)
    return out["exp_id"]


def save_predictions(name: str, p, cfg: dict) -> Path:
    """Guarda las predicciones en dev de una configuración final (para la Etapa 4)."""
    d = _resolve(cfg["results"]["predictions_dir"])
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{name}.npy"
    np.save(f, _arr(p).astype("float32"))
    return f

def evaluate_and_log(y_dev, p_dev, cfg: dict, *, stage: str, model: str, variant: str = "",
                     cost_strategy: str = "", seed="", hyperparams=None, n_features="",
                     best_iter="", train_time_s="", y_train=None, p_train=None,
                     notes: str = "") -> dict:
    """Evalúa en dev, registra la fila en results.csv y devuelve las métricas (con exp_id)."""
    m = evaluate_scores(y_dev, p_dev, cfg)
    row = {
        "stage": stage, "model": model, "variant": variant, "cost_strategy": cost_strategy,
        "seed": seed, "hyperparams": hyperparams or {}, "n_features": n_features,
        "best_iter": best_iter, "train_time_s": train_time_s, "notes": notes,
        "pr_auc_train": pr_auc(y_train, p_train) if p_train is not None else "",
        **{k: m[k] for k in ["pr_auc", "roc_auc", "recall_at_p", "f1_max", "min_cost", "min_cost_rel"]},
    }
    m["exp_id"] = log_result(row, cfg["results"]["path"])
    return m

def paired_bootstrap_diff(y, p_a, p_b, n: int = 200, seed: int = 19):
    """Diferencia de PR-AUC (b - a) e intervalo del 95% por bootstrap pareado en dev.

    Remuestrea transacciones de forma independiente: con deriva temporal subestima
    algo la incertidumbre real. Sirve para descartar ruido, no para probar mejoras pequeñas.
    """
    y, p_a, p_b = _arr(y).astype(int), _arr(p_a), _arr(p_b)
    rng = np.random.default_rng(seed)
    N = len(y)
    d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        d[i] = average_precision_score(y[idx], p_b[idx]) - average_precision_score(y[idx], p_a[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)