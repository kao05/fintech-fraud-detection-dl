"""Carga, unión y guardado de los datos IEEE-CIS. Rutas relativas a la raíz del proyecto."""
import re
from pathlib import Path

import pandas as pd

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