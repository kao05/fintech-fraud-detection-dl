import numpy as np
import pytest

pytest.importorskip("keras")
from src.models import build_mlp
from src.train import _dev_pr_auc_callback, train_model


def test_callback_pr_auc_informa_si_predicciones_no_finitas():
    callback = _dev_pr_auc_callback(np.ones((2, 1), dtype="float32"), np.array([0, 1]))

    class Model:
        def predict(self, *args, **kwargs):
            return np.array([[np.nan], [0.5]], dtype="float32")

    callback.set_model(Model())
    with pytest.raises(FloatingPointError, match="predicciones no finitas"):
        callback.on_epoch_end(0)


def test_entrenamiento_focal_aplica_clipnorm_global():
    X = np.random.default_rng(0).normal(size=(32, 12)).astype("float32")
    y = np.array([0, 1] * 16, dtype="float32")
    cfg = {
        "nn": {"common": {"learning_rate": 1e-3, "batch_size": 8, "max_epochs": 1,
                          "early_stopping_patience": 1}},
        "imbalance": {"class_weight_mode": "balanced", "focal_gamma": 2.0, "focal_alpha": None},
        "cost": {"c_FN": 10, "c_FP": 1, "target_precision": 0.5},
    }

    model = build_mlp(X.shape[1], hidden=(8,))
    train_model(model, (X[:24], y[:24]), (X[24:], y[24:]), cfg, "focal", 19,
                kind="mlp", log=False, save_weights=False)

    assert model.optimizer.global_clipnorm == 1.0
