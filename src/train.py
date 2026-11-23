import numpy as np


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