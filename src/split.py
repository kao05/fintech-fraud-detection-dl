import numpy as np
import pandas as pd


def temporal_split(df: pd.DataFrame, cfg: dict):
    """Split temporal por VALOR de tiempo. Única definición del split del proyecto.

    Ordena por TransactionDT, calcula los cortes de tiempo en las fracciones
    acumuladas (train / train+dev) y asigna cada fila según su valor de tiempo.
    Así dos filas con el mismo TransactionDT nunca quedan en splits distintos.
    """
    time_col = cfg["columns"]["time"]
    frac = cfg["split"]

    df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    t = df[time_col].to_numpy(dtype="float64")
    n = len(df)

    # cortes de tiempo: el valor de TransactionDT en la posición de cada fracción
    t_cut_dev = t[int(n * frac["train"])]
    t_cut_test = t[int(n * (frac["train"] + frac["dev"]))]

    train = df[df[time_col] < t_cut_dev]
    dev = df[(df[time_col] >= t_cut_dev) & (df[time_col] < t_cut_test)]
    test = df[df[time_col] >= t_cut_test]

    check_split(train, dev, test, cfg)
    return train, dev, test


def check_split(train, dev, test, cfg: dict) -> None:
    """Verifica que los splits sean disjuntos, completos y estrictamente ordenados en el tiempo."""
    tc, ic = cfg["columns"]["time"], cfg["columns"]["id"]

    assert train[tc].max() < dev[tc].min(), "Solapamiento de tiempo entre train y dev"
    assert dev[tc].max() < test[tc].min(), "Solapamiento de tiempo entre dev y test"

    ids = [set(s[ic]) for s in (train, dev, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), \
        "TransactionID repetido entre splits"


def split_summary(train, dev, test, cfg: dict) -> pd.DataFrame:
    """Tabla con tamaño, fraudes, tasa y rango de tiempo (en días) de cada split."""
    tc, yc = cfg["columns"]["time"], cfg["columns"]["target"]
    rows = []
    for name, part in (("train", train), ("dev", dev), ("test", test)):
        rows.append({
            "split": name,
            "filas": len(part),
            "pct_filas": 100 * len(part) / (len(train) + len(dev) + len(test)),
            "fraudes": int(part[yc].sum()),
            "tasa_fraude_%": 100 * part[yc].mean(),
            "dia_inicio": part[tc].min() / 86400,
            "dia_fin": part[tc].max() / 86400,
        })
    return pd.DataFrame(rows).set_index("split")