"""Búsqueda aleatoria de hiperparámetros de las redes (Etapa 3, Acción 8).

Todo es reproducible: las configuraciones dependen solo de la semilla de búsqueda y de la
arquitectura (n mayor = las mismas primeras n configuraciones + las nuevas). Cada corrida
deja una fila en results.csv; si el notebook se interrumpe, al relanzar se saltan las ya hechas.

Fases (todas con class_weight salvo la focal):
  A   búsqueda con seeds[0], n_trials configuraciones por arquitectura
  A2  las n_finalists mejores por arquitectura se repiten con las demás semillas
      -> la configuración final es la de mejor media sobre las 3 semillas
  F   búsqueda de (gamma, alpha) de la focal con la mejor arquitectura (seeds[0])
  F2  la mejor (gamma, alpha) se repite en las 4 arquitecturas con las 3 semillas
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KINDS = ["mlp", "mlp_regularizado", "mlp_embeddings", "residual_mlp"]


# ------------------------------------------------------------------ muestreo
def _hidden(depth: int, width: int, shape: str) -> list[int]:
    if shape == "constant":
        return [width] * depth
    return [max(16, width // 2 ** i) for i in range(depth)]      # embudo: width, width/2, ...


def sample_trials(kind: str, cfg: dict, n: int) -> list[dict]:
    """n configuraciones distintas y reproducibles para una arquitectura."""
    s = cfg["nn"]["search"]
    sp = s["space"]
    rng = np.random.default_rng([int(s["seed"]), KINDS.index(kind)])
    trials, seen, attempts = [], set(), 0
    while len(trials) < n and attempts < 100 * max(n, 1):
        attempts += 1
        t = {"train": {"learning_rate": float(rng.choice(sp["learning_rate"])),
                       "batch_size": int(rng.choice(sp["batch_size"]))},
             "arch": {}}
        if kind == "residual_mlp":
            t["arch"] = {"width": int(rng.choice(sp["residual_width"])),
                         "n_blocks": int(rng.choice(sp["n_blocks"]))}
        else:
            t["arch"] = {"hidden": _hidden(int(rng.choice(sp["depth"])),
                                           int(rng.choice(sp["width"])),
                                           str(rng.choice(sp["shape"])))}
        if kind != "mlp":                                          # la MLP convencional no se regulariza
            t["arch"]["dropout"] = float(rng.choice(sp["dropout"]))
            t["arch"]["l2"] = float(rng.choice(sp["l2"]))
        if kind == "mlp_embeddings":
            t["arch"]["emb"] = {**cfg["nn"]["embedding_dim"], "cap": int(rng.choice(sp["emb_cap"]))}
        key = json.dumps(t, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        t["id"] = f"cw_{len(trials):02d}"
        trials.append(t)
    return trials


def describe(t: dict) -> str:
    a, tr = t["arch"], t["train"]
    if "hidden" in a:
        s = f"hidden={a['hidden']}"
    else:
        s = f"width={a['width']} bloques={a['n_blocks']}"
    if "dropout" in a:
        s += f" drop={a['dropout']} l2={a['l2']:g}"
    if "emb" in a:
        s += f" emb_cap={a['emb']['cap']}"
    return f"{s} lr={tr['learning_rate']:g} bs={tr['batch_size']}"


def focal_grid(cfg: dict) -> list[dict]:
    f = cfg["nn"]["search"]["focal"]
    return [{"gamma": float(g), "alpha": None if a is None else float(a)}
            for g in f["gamma"] for a in f["alpha"]]


def focal_variant(cw_id: str, f: dict) -> str:
    a = "none" if f["alpha"] is None else f"{f['alpha']:g}"
    return f"focal_{cw_id}_g{f['gamma']:g}_a{a}"


# ------------------------------------------------------------------ resultados
def read_results(cfg: dict) -> pd.DataFrame:
    p = ROOT / cfg["results"]["path"]
    if not p.exists():
        return pd.DataFrame(columns=["stage", "model", "variant", "cost_strategy", "seed", "pr_auc"])
    r = pd.read_csv(p)
    r = r[r["stage"] == "nn"].copy()
    r["seed"] = pd.to_numeric(r["seed"], errors="coerce")
    r["variant"] = r["variant"].astype(str)
    return r.dropna(subset=["seed"]).drop_duplicates(["model", "variant", "cost_strategy", "seed"], keep="last")


def done_runs(cfg: dict) -> set:
    r = read_results(cfg)
    return set(zip(r["model"], r["variant"], r["cost_strategy"], r["seed"].astype(int)))


def seed_table(cfg: dict, kind: str, strategy: str, prefix: str) -> pd.DataFrame:
    """Media y desviación del PR-AUC en dev por configuración, solo con las 3 semillas completas."""
    r = read_results(cfg)
    r = r[(r["model"] == kind) & (r["cost_strategy"] == strategy) & r["variant"].str.startswith(prefix)]
    g = r.groupby("variant")["pr_auc"].agg(mean="mean", sd="std", n="count").reset_index()
    g = g[g["n"] == len(cfg["seeds"])]
    return g.sort_values("mean", ascending=False).reset_index(drop=True)


def top_configs(cfg: dict, kind: str, k: int) -> list[str]:
    r = read_results(cfg)
    r = r[(r["model"] == kind) & (r["cost_strategy"] == "class_weight")
          & (r["seed"] == cfg["seeds"][0]) & r["variant"].str.startswith("cw_")]
    return list(r.sort_values("pr_auc", ascending=False)["variant"].head(k))


# ------------------------------------------------------------------ ejecución
def make_context(cfg: dict, tr: dict, dv: dict, vocab: dict) -> dict:
    idx_cols = list(vocab)
    return {"cfg": cfg, "vocab": vocab, "idx_cols": idx_cols, "n_features": tr["X"].shape[1],
            "y_tr": tr["y"]["isFraud"], "y_dv": dv["y"]["isFraud"], "_cache": {},
            "_raw": {"tr": (tr["X"], tr["idx"][idx_cols]), "dv": (dv["X"], dv["idx"][idx_cols])}}


def _inputs(ctx: dict, kind: str):
    from src.train import make_inputs
    k = "mlp_embeddings" if kind == "mlp_embeddings" else "mlp"
    if k not in ctx["_cache"]:
        ctx["_cache"][k] = tuple(make_inputs(X, idx, k, ctx["idx_cols"])
                                 for X, idx in (ctx["_raw"]["tr"], ctx["_raw"]["dv"]))
    return ctx["_cache"][k]


def run_one(ctx: dict, kind: str, trial: dict, strategy: str, seed: int, *,
            variant: str, focal: dict | None = None, notes: str = "") -> dict:
    import keras
    from src.models import build_model
    from src.train import train_model

    cfg = ctx["cfg"]
    keras.backend.clear_session()
    gc.collect()
    keras.utils.set_random_seed(seed)          # ANTES de construir: la inicialización de pesos depende de la semilla
    model = build_model(kind, cfg, ctx["n_features"], ctx["vocab"], **trial["arch"])
    x_tr, x_dv = _inputs(ctx, kind)
    kw = {"focal_gamma": focal["gamma"], "focal_alpha": focal["alpha"]} if strategy == "focal" else {}
    return train_model(model, (x_tr, ctx["y_tr"]), (x_dv, ctx["y_dv"]), cfg, strategy, seed,
                       kind=kind, variant=variant, hyperparams=trial["arch"],
                       learning_rate=trial["train"]["learning_rate"],
                       batch_size=trial["train"]["batch_size"], n_features=ctx["n_features"],
                       save_preds=True, notes=notes, **kw)


def _report(kind, vid, trial, seed, out, extra=""):
    m = out["metrics"]
    print(f"{kind:<17}{vid:<34} s{seed:<3}| {describe(trial)}{extra} | época {out['best_epoch']:>3} | "
          f"PR-AUC dev {m['pr_auc']:.4f} | {out['train_time_s'] / 60:.1f} min")


def _trials_by_id(cfg: dict, kind: str) -> dict:
    return {t["id"]: t for t in sample_trials(kind, cfg, cfg["nn"]["search"]["n_trials"])}


def run_search(ctx: dict, kind: str, n_trials: int | None = None) -> None:
    """Fase A: n_trials configuraciones con class_weight y seeds[0]."""
    cfg, seed = ctx["cfg"], ctx["cfg"]["seeds"][0]
    n = n_trials or cfg["nn"]["search"]["n_trials"]
    done, t0 = done_runs(cfg), time.perf_counter()
    for t in sample_trials(kind, cfg, n):
        if (kind, t["id"], "class_weight", seed) in done:
            print(f"{kind} {t['id']}: ya corrida, se omite")
            continue
        _report(kind, t["id"], t, seed, run_one(ctx, kind, t, "class_weight", seed,
                                                variant=t["id"], notes="Fase A: búsqueda"))
    print(f"Tanda terminada en {(time.perf_counter() - t0) / 60:.1f} min")


def run_finalists(ctx: dict, kind: str, k: int | None = None) -> None:
    """Fase A2: las k mejores configuraciones se repiten con las demás semillas."""
    cfg = ctx["cfg"]
    k = k or cfg["nn"]["search"]["n_finalists"]
    trials, done, t0 = _trials_by_id(cfg, kind), done_runs(cfg), time.perf_counter()
    for vid in top_configs(cfg, kind, k):
        for s in cfg["seeds"][1:]:
            if (kind, vid, "class_weight", s) in done:
                print(f"{kind} {vid} s{s}: ya corrida, se omite")
                continue
            _report(kind, vid, trials[vid], s, run_one(ctx, kind, trials[vid], "class_weight", s,
                                                       variant=vid, notes="Fase A2: finalista"))
    print(f"Tanda terminada en {(time.perf_counter() - t0) / 60:.1f} min")


def best_cw(cfg: dict, kind: str) -> str:
    t = seed_table(cfg, kind, "class_weight", "cw_")
    if t.empty:
        raise RuntimeError(f"{kind}: aún no hay configuraciones con las {len(cfg['seeds'])} semillas")
    return t.loc[0, "variant"]


def run_focal_search(ctx: dict, kind: str) -> None:
    """Fase F: rejilla (gamma, alpha) con la mejor arquitectura, su mejor config y seeds[0]."""
    cfg, seed = ctx["cfg"], ctx["cfg"]["seeds"][0]
    cw_id = best_cw(cfg, kind)
    trial, done, t0 = _trials_by_id(cfg, kind)[cw_id], done_runs(cfg), time.perf_counter()
    for f in focal_grid(cfg):
        v = focal_variant(cw_id, f)
        if (kind, v, "focal", seed) in done:
            print(f"{v}: ya corrida, se omite")
            continue
        _report(kind, v, trial, seed, run_one(ctx, kind, trial, "focal", seed, variant=v, focal=f,
                                              notes="Fase F: búsqueda de gamma y alpha"))
    print(f"Tanda terminada en {(time.perf_counter() - t0) / 60:.1f} min")


def best_focal(cfg: dict, kind: str, cw_id: str):
    r = read_results(cfg)
    r = r[(r["model"] == kind) & (r["cost_strategy"] == "focal") & (r["seed"] == cfg["seeds"][0])]
    best, score = None, -1.0
    for f in focal_grid(cfg):
        row = r[r["variant"] == focal_variant(cw_id, f)]
        if len(row) and float(row["pr_auc"].iloc[-1]) > score:
            best, score = f, float(row["pr_auc"].iloc[-1])
    return best, score


def run_focal_finals(ctx: dict, f_best: dict) -> None:
    """Fase F2: la mejor (gamma, alpha) en las 4 arquitecturas con las 3 semillas."""
    cfg, done, t0 = ctx["cfg"], done_runs(ctx["cfg"]), time.perf_counter()
    for kind in KINDS:
        cw_id = best_cw(cfg, kind)
        trial, v = _trials_by_id(cfg, kind)[cw_id], focal_variant(cw_id, f_best)
        for s in cfg["seeds"]:
            if (kind, v, "focal", s) in done:
                print(f"{kind} {v} s{s}: ya corrida, se omite")
                continue
            _report(kind, v, trial, s, run_one(ctx, kind, trial, "focal", s, variant=v, focal=f_best,
                                               notes="Fase F2: configuración final focal"))
    print(f"Tanda terminada en {(time.perf_counter() - t0) / 60:.1f} min")


# ------------------------------------------------------------------ cierre
def final_matrix(cfg: dict, f_best: dict):
    """Tabla final (arquitectura x estrategia, media ± sd sobre las semillas) y configuraciones finales."""
    r = read_results(cfg)
    rows, finals = [], {}
    for kind in KINDS:
        cw_id = best_cw(cfg, kind)
        trial = _trials_by_id(cfg, kind)[cw_id]
        for strat, vid, focal in [("class_weight", cw_id, None),
                                  ("focal", focal_variant(cw_id, f_best), f_best)]:
            sub = r[(r["model"] == kind) & (r["cost_strategy"] == strat) & (r["variant"] == vid)]
            sub = sub.sort_values("seed")
            if len(sub) != len(cfg["seeds"]):
                raise RuntimeError(f"{kind}/{strat}/{vid}: {len(sub)} de {len(cfg['seeds'])} semillas")
            per_seed = {int(s): float(p) for s, p in zip(sub["seed"], sub["pr_auc"])}
            rows.append({"arquitectura": kind, "estrategia": strat, "variante": vid,
                         "PR-AUC media": sub["pr_auc"].mean(), "sd": sub["pr_auc"].std(),
                         "ROC-AUC media": sub["roc_auc"].mean(),
                         "recall@P media": sub["recall_at_p"].mean(),
                         "min_cost_rel media": sub["min_cost_rel"].mean(),
                         **{f"s{s}": p for s, p in per_seed.items()}})
            finals.setdefault(kind, {})[strat] = {
                "variant": vid, "arch": trial["arch"], "train": trial["train"], "focal": focal,
                "seeds": list(per_seed), "pr_auc_per_seed": per_seed,
                "weights_pattern": f"models/nn/{kind}_{strat}_{vid}_s<seed>.weights.h5",
                "dev_preds_pattern": f"reports/preds/{kind}_{strat}_{vid}_s<seed>_dev.npy"}
    return pd.DataFrame(rows), finals


def save_final_configs(finals: dict, path: str = "reports/nn_final_configs.json") -> Path:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(finals, indent=2), encoding="utf-8")
    return p