"""Carga, unión y guardado de los datos IEEE-CIS. Rutas relativas a la raíz del proyecto."""
import re
from pathlib import Path

import pandas as pd

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 y orden temporal."""
    f64 = df.select_dtypes("float64").columns
    df[f64] = df[f64].astype("float32")
    return df.sort_values(TIME_COL).reset_index(drop=True)


def load_raw(raw_dir: Path = RAW_DIR):
    """Lee los CSV de train (el test no tiene etiquetas)."""
    trans = pd.read_csv(raw_dir / "train_transaction.csv")
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    return trans, ident


def merge_tables(trans: pd.DataFrame, ident: pd.DataFrame) -> pd.DataFrame:
    """Left join por TransactionID + bandera has_identity."""
    df = trans.merge(ident, on=ID_COL, how="left")
    df["has_identity"] = df[ID_COL].isin(ident[ID_COL]).astype("int8")
    return _finalize(df)


def save_processed(df: pd.DataFrame, name: str = "train_merged.parquet") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_parquet(path, index=False)
    return path


def load_processed(name: str = "train_merged.parquet") -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / name)
    if "has_identity" not in df.columns:  # parquet guardado antes de añadirla
        id_cols = [c for c in df.columns if c.startswith("id_")] + ["DeviceType", "DeviceInfo"]
        df["has_identity"] = df[id_cols].notna().any(axis=1).astype("int8")
    return _finalize(df)


def column_groups(df: pd.DataFrame) -> dict:
    """Familias de columnas (base del diccionario de data/README.md)."""
    cols = list(df.columns)
    pick = lambda pat: [c for c in cols if re.fullmatch(pat, c)]
    return {
        "card": pick(r"card\d"),
        "addr": pick(r"addr\d"),
        "dist": pick(r"dist\d"),
        "email": pick(r"[PR]_emaildomain"),
        "C": pick(r"C\d+"),
        "D": pick(r"D\d+"),
        "M": pick(r"M\d+"),
        "V": pick(r"V\d+"),
        "id": pick(r"id_\d+"),
        "device": pick(r"Device(Type|Info)"),
    }

# Funcion para mandar cargar la configuracion igual a la que tenemos con los datos 
def load_config(path: str = "configs/default.yaml") -> dict:
    """Lee configs/default.yaml (ruta relativa a la raíz del proyecto) y la valida."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)
    return cfg

def _validate_config(cfg: dict) -> None:
    s = cfg["split"]
    assert abs(s["train"] + s["dev"] + s["test"] - 1.0) < 1e-9, "Las fracciones del split deben sumar 1"
    assert all(isinstance(x, int) for x in cfg["seeds"]) and len(cfg["seeds"]) >= 1, "seeds: lista de enteros"
    c = cfg["cost"]
    assert c["c_FN"] > 0 and c["c_FP"] > 0, "Los costos deben ser positivos"
    assert 0 < c["target_precision"] < 1, "target_precision debe estar en (0, 1)"
    p = cfg["preprocessing"]
    assert 0 < p["corr_threshold"] < 1, "corr_threshold debe estar en (0, 1)"
    assert cfg["preprocessing"]["add_null_indicators"] is True, "Regla 7: no imputar sin indicador de nulo"
    assert cfg["imbalance"]["resample"] is False, "Regla 3: no se remuestrea dev ni test"