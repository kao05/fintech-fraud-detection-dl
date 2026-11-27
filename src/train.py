import numpy as np
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CFG = object()   # marca: "toma este valor de la config"

def compute_class_weights(y_train, mode: str = "balanced") -> dict:
    """Pesos de clase calculados SOLO con y_train (nunca con dev ni test).

    class_weight     : {0: w0, 1: w1}, para Keras (class_weight=...) y scikit-learn.
    scale_pos_weight : n0 / n1, el equivalente para XGBoost y LightGBM.
    """
    if mode != "balanced":
        raise ValueError(f"class_weight_mode no soportado: {mode}")
    y = np.asarray(y_train)
    n, n1 = len(y), int((y == 1).sum())
    n0 = n - n1
    return {
        "class_weight": {0: n / (2 * n0), 1: n / (2 * n1)},
        "scale_pos_weight": n0 / n1,
        "n0": n0,
        "n1": n1,
    }

def make_inputs(X, idx=None, kind: str = "mlp", idx_cols=None):
    """Entradas del modelo: un arreglo para las MLP tabulares; un diccionario para embeddings."""
    x = np.asarray(X, dtype="float32")
    if kind != "mlp_embeddings":
        return x
    d = {"x": x}
    for c in (idx_cols or list(idx.columns)):
        d[c] = idx[c].to_numpy("int32").reshape(-1, 1)
    return d


def initial_output_bias(y_train, strategy: str) -> float:
    """Con class_weight los pesos ya equilibran las clases (sesgo 0); con focal, log(n1/n0)."""
    if strategy != "focal":
        return 0.0
    y = np.asarray(y_train)
    n1 = int((y == 1).sum())
    return float(np.log(n1 / (len(y) - n1)))


def _dev_pr_auc_callback(x_dev, y_dev, batch_size: int = 8192):
    """Calcula el PR-AUC exacto en dev al final de cada época (lo usa el early stopping)."""
    import keras
    from sklearn.metrics import average_precision_score

    class DevPRAUC(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.scores = []

        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(x_dev, batch_size=batch_size, verbose=0).ravel()
            s = float(average_precision_score(y_dev, p))
            self.scores.append(s)
            if logs is not None:
                logs["val_pr_auc"] = s

    return DevPRAUC()


def train_model(model, train, dev, cfg, strategy, seed, *, kind, variant="", hyperparams=None,
                learning_rate=None, batch_size=None, max_epochs=None, patience=None,
                focal_gamma=None, focal_alpha=_CFG, n_features="", stage="nn",
                log=True, save_weights=True, save_preds=False, notes=""):
    """Entrena una red con el protocolo del proyecto.

    train, dev      : (entradas, y) ya construidas con make_inputs.
    strategy        : 'class_weight' (entropía cruzada + pesos de train) o 'focal'
                      (BinaryFocalCrossentropy, sin pesos para no contarlos dos veces).
    Early stopping  : PR-AUC exacto en dev, restore_best_weights=True.
    Devuelve        : dict con p_dv, métricas, historial, mejor época y exp_id.
    Llamar a keras.backend.clear_session() ANTES de construir cada modelo.
    """
    import keras
    from sklearn.metrics import average_precision_score  # noqa: F401  (usado en el callback)
    from src.evaluate import evaluate_and_log, save_predictions

    if strategy not in ("class_weight", "focal"):
        raise ValueError(f"estrategia no soportada: {strategy}")

    nn, imb = cfg["nn"]["common"], cfg["imbalance"]
    lr = learning_rate or nn["learning_rate"]
    bs = batch_size or nn["batch_size"]
    epochs = max_epochs or nn["max_epochs"]
    pat = patience or nn["early_stopping_patience"]
    gamma = focal_gamma if focal_gamma is not None else imb["focal_gamma"]
    alpha = imb["focal_alpha"] if focal_alpha is _CFG else focal_alpha

    (x_tr, y_tr), (x_dv, y_dv) = train, dev
    y_tr_a, y_dv_a = np.asarray(y_tr, dtype="float32"), np.asarray(y_dv, dtype="float32")

    keras.utils.set_random_seed(seed)
    model.get_layer("fraud_prob").bias.assign(
        np.array([initial_output_bias(y_tr_a, strategy)], dtype="float32"))

    if strategy == "class_weight":
        loss = keras.losses.BinaryCrossentropy()
        class_weight = compute_class_weights(y_tr_a, imb["class_weight_mode"])["class_weight"]
    else:
        loss = keras.losses.BinaryFocalCrossentropy(
            gamma=gamma, alpha=alpha if alpha is not None else 0.25,
            apply_class_balancing=alpha is not None)
        class_weight = None

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)

    dev_cb = _dev_pr_auc_callback(x_dv, y_dv_a)
    callbacks = [dev_cb,   # debe ir ANTES del early stopping: escribe val_pr_auc en los logs
                 keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=pat,
                                               restore_best_weights=True)]
    t0 = time.perf_counter()
    hist = model.fit(x_tr, y_tr_a, batch_size=bs, epochs=epochs, validation_data=(x_dv, y_dv_a),
                     class_weight=class_weight, callbacks=callbacks, verbose=0)
    t = time.perf_counter() - t0

    best_epoch = int(np.argmax(dev_cb.scores)) + 1
    p_dv = model.predict(x_dv, batch_size=8192, verbose=0).ravel()
    p_tr = model.predict(x_tr, batch_size=8192, verbose=0).ravel()

    history = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    history["val_pr_auc"] = [float(s) for s in dev_cb.scores]

    hp = {"arch": kind, **(hyperparams or {}), "learning_rate": lr, "batch_size": bs,
          "strategy": strategy, "epochs_run": len(dev_cb.scores), "patience": pat}
    if strategy == "focal":
        hp.update({"focal_gamma": gamma, "focal_alpha": alpha})

    tag = f"{kind}_{strategy}_{variant or 'base'}_s{seed}"
    if save_weights:
        d = ROOT / cfg["paths"]["models_dir"] / "nn"
        d.mkdir(parents=True, exist_ok=True)
        model.save_weights(d / f"{tag}.weights.h5")
        (d / f"{tag}_history.json").write_text(json.dumps(history))
    if save_preds:
        save_predictions(tag + "_dev", p_dv, cfg)

    out = {"p_dv": p_dv, "history": history, "best_epoch": best_epoch, "train_time_s": t, "tag": tag}
    if log:
        out["metrics"] = evaluate_and_log(
            y_dv_a, p_dv, cfg, stage=stage, model=kind, variant=variant, cost_strategy=strategy,
            seed=seed, hyperparams=hp, n_features=n_features, best_iter=best_epoch,
            train_time_s=round(t, 1), y_train=y_tr_a, p_train=p_tr, notes=notes)
        out["exp_id"] = out["metrics"]["exp_id"]
    else:
        from src.evaluate import evaluate_scores
        out["metrics"] = evaluate_scores(y_dv_a, p_dv, cfg)
    return out