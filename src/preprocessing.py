"""Preprocesamiento. Parte 1: limpieza (Acción 2) y variables derivadas (Acción 3).

Nada de esta parte se ajusta con train: son transformaciones fila a fila, así que
se aplican igual a train, dev y test sin riesgo de fuga. Las partes que sí se
ajustan con train (Acciones 4 a 8) se añaden más abajo como clases con fit/transform.
"""
import numpy as np
import pandas as pd
import hashlib
import re
from scipy.stats import rankdata
import json
import joblib
from pathlib import Path
from src.data import ROOT, load_config

DERIVED_COLS = ["hour_sin", "hour_cos", "amt_log", "amt_cents"]


def derived_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Acción 3. Devuelve solo las 4 columnas derivadas (mismo índice que df)."""
    c, d = cfg["columns"], cfg["derived"]
    time = df[c["time"]].to_numpy(dtype="float64")
    amt = df[c["amount"]].to_numpy(dtype="float64")   # float64: los centavos exigen precisión
    assert (amt >= 0).all(), "log1p exige importes >= 0"

    # Hora relativa del día (el desfase de TransactionDT es desconocido) y su codificación cíclica
    hour = (time // d["seconds_per_hour"]) % d["hour_period"]
    angle = 2 * np.pi * hour / d["hour_period"]

    out = pd.DataFrame(index=df.index)
    out["hour_sin"] = np.sin(angle).astype("float32")
    out["hour_cos"] = np.cos(angle).astype("float32")
    out["amt_log"] = np.log1p(amt).astype("float32")
    # Centavos 0-99: se redondea a centavo entero primero para evitar errores de coma flotante
    out["amt_cents"] = (np.round(amt * 100).astype("int64") % 100).astype("float32")
    return out


def prepare_frame(df: pd.DataFrame, cfg: dict):
    """Acciones 2 y 3. Separa un split en (X, y, meta) y añade las variables derivadas.

    X    : predictores (incluye has_identity y las 4 derivadas; sin id, tiempo crudo ni importe crudo)
    y    : isFraud (int8)
    meta : TransactionID, TransactionDT y TransactionAmt crudos (no son variables; sirven para
           el test anti-fuga, el costo ponderado por importe y el análisis de falsos negativos)
    No se deduplican filas (A8).
    """
    c = cfg["columns"]
    derived = derived_features(df, cfg)

    meta = df[[c["id"], c["time"], c["amount"]]].reset_index(drop=True)
    y = df[c["target"]].astype("int8").reset_index(drop=True)

    to_drop = sorted(set(c["drop"]) | {c["target"], c["amount"]})
    X = df.drop(columns=to_drop)
    X.index = pd.RangeIndex(len(X))
    for col in DERIVED_COLS:
        X[col] = derived[col].to_numpy()

    assert X.columns.is_unique, "Columnas duplicadas en X"
    assert not set(to_drop) & set(X.columns), "Quedó una columna prohibida en X"
    assert all(k in X.columns for k in c["keep_flags"]), "Falta una bandera que debía conservarse"
    assert not X[DERIVED_COLS].isna().any().any(), "Hay NaN en las variables derivadas"
    return X, y, meta

"""Acción 4. Un indicador de nulo por PATRÓN de ausencia, no por columna.
fit(X_train): agrupa las columnas con nulos que tienen exactamente el mismo patrón deausencia a lo largo de las filas de train y crea un grupo por patrón.
transform(X): devuelve un DataFrame solo con los indicadores (int8): 1 si el valor de la columna representante del grupo falta en esa fila.
Debe aplicarse ANTES de imputar (Acción 5), porque después ya no hay nulos que detectar.
"""
class NullPatternIndicators:
    def __init__(self, prefix: str = "null_pat_", min_rows: int = 0):
        self.prefix = prefix
        self.min_rows = min_rows
        self.groups_: dict[str, list[str]] = {}
        self.rep_: dict[str, str] = {}
        self.train_null_rate_: dict[str, float] = {}

    def fit(self, X_train: pd.DataFrame):
        isna = X_train.isna()
        by_signature: dict[str, list[str]] = {}
        for col in X_train.columns:            # orden de columnas: asignación determinista de nombres
            v = isna[col].to_numpy()
            if not v.any() or v.all():          # sin nulos o 100% nula en train: no aportan
                continue
            n_null = int(v.sum())
            if n_null < self.min_rows or (len(v) - n_null) < self.min_rows:
                continue                    # patrón sin filas suficientes para aprenderse
            sig = hashlib.md5(np.packbits(v).tobytes()).hexdigest()
            by_signature.setdefault(sig, []).append(col)

        self.groups_, self.rep_, self.train_null_rate_ = {}, {}, {}
        for i, cols in enumerate(by_signature.values()):
            name = f"{self.prefix}{i}"
            self.groups_[name] = cols
            self.rep_[name] = cols[0]
            self.train_null_rate_[name] = float(isna[cols[0]].mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for name, rep in self.rep_.items():
            out[name] = X[rep].isna().astype("int8")
        return out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)

    # ----- utilidades de diagnóstico (no participan en el ajuste) -----
    def coherence(self, X: pd.DataFrame) -> pd.Series:
        """Por grupo: fracción de columnas del grupo cuyo patrón de nulos en X coincide con el de
        la representante. 1.0 = el patrón de train se conserva tal cual en este split."""
        res = {}
        for name, cols in self.groups_.items():
            rep = X[self.rep_[name]].isna().to_numpy()
            res[name] = float(np.mean([(X[c].isna().to_numpy() == rep).all() for c in cols]))
        return pd.Series(res, name="coherencia")

    def uncovered_null_columns(self, X: pd.DataFrame) -> list[str]:
        """Columnas con nulos en X que no pertenecen a ningún grupo de train (sin indicador)."""
        covered = {c for cols in self.groups_.values() for c in cols}
        has_null = X.columns[X.isna().any()]
        return [c for c in has_null if c not in covered]

    def summary(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Tabla por grupo con tasa de fraude cuando falta y cuando no falta (usar con train)."""
        ind = self.transform(X)
        rows = []
        for name, cols in self.groups_.items():
            m = ind[name].astype(bool).to_numpy()
            yy = y.to_numpy()
            rows.append({
                "grupo": name,
                "n_cols": len(cols),
                "ejemplo": ", ".join(cols[:3]),
                "nulo_%": 100 * m.mean(),
                "fraude_si_nulo_%": 100 * yy[m].mean() if m.any() else np.nan,
                "fraude_si_presente_%": 100 * yy[~m].mean() if (~m).any() else np.nan,
            })
        out = pd.DataFrame(rows).set_index("grupo")
        out["dif_pp"] = out["fraude_si_nulo_%"] - out["fraude_si_presente_%"]
        return out


"""Acción 5. Imputación ajustada solo con train.
Numéricas  : mediana de train (o 0, según cfg["preprocessing"]["numeric_impute"]).
Categóricas: texto y los códigos numéricos de cfg["columns"]["categorical_numeric_coded"]
                pasan a texto, y el nulo se convierte en la categoría explícita "missing".
Se aplica DESPUÉS de los indicadores de nulo (Acción 4): sin ellos, imputar borraría la señal.
"""
class Imputer:
    def __init__(self, cfg: dict):
        p = cfg["preprocessing"]
        self.strategy = p["numeric_impute"]
        self.missing_label = p["categorical_missing_label"]
        self.coded = list(cfg["columns"]["categorical_numeric_coded"])
        self.fill_: dict[str, float] = {}
        self.num_cols_: list[str] = []
        self.cat_cols_: list[str] = []

    def fit(self, X_train: pd.DataFrame):
        self.cat_cols_ = [c for c in X_train.columns
                          if c in self.coded or not pd.api.types.is_numeric_dtype(X_train[c])]
        self.num_cols_ = [c for c in X_train.columns if c not in self.cat_cols_]
        if self.strategy == "median":
            med = X_train[self.num_cols_].median()
            # una columna 100% nula en train no tendría mediana: se rellena con 0
            self.fill_ = med.fillna(0.0).astype("float64").to_dict()
        elif self.strategy == "zero":
            self.fill_ = {c: 0.0 for c in self.num_cols_}
        else:
            raise ValueError(f"numeric_impute no soportado: {self.strategy}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.num_cols_:
            if X[c].isna().any():
                X[c] = X[c].fillna(np.float32(self.fill_[c]))
        for c in self.cat_cols_:
            s = X[c]
            if pd.api.types.is_numeric_dtype(s):
                s = s.round().astype("Int64")          # 13926.0 -> "13926" (sin ".0")
            X[c] = s.astype("string").fillna(self.missing_label).astype(object)
        assert not X.isna().any().any(), "Quedaron NaN tras imputar"
        return X

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)

def _safe(s) -> str:
    """Nombre de columna seguro (XGBoost rechaza [, ] y <; además evita espacios y puntos)."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(s)).strip("_") or "empty"


"""Acción 6. Codificación de categóricas. Vocabularios y frecuencias salen SOLO de train.
    - Baja cardinalidad (<= onehot_max_cardinality categorías distintas en train): one-hot.
    - Alta cardinalidad: (a) índice entero para embeddings (0 = 'desconocido') y
                         (b) frecuencia relativa de train para los modelos tabulares.
    - Categorías con frecuencia < min_category_freq o nuevas en dev/test -> 'desconocido'.
    - Importe exacto (hallazgo C3): amt_freq (frecuencia del valor exacto en train; nuevo -> 0) y
      amt_price_point_idx (índice del precio recurrente; el resto y los nuevos -> 0 = 'otros').
"""
class CategoricalEncoder:
    def __init__(self, cfg: dict):
        p = cfg["preprocessing"]
        self.onehot_max = p["onehot_max_cardinality"]
        self.min_freq = p["min_category_freq"]
        self.price_min_freq = p["amt_price_point_min_freq"]
        self.unknown = p["unknown_label"]
        self.cols_, self.low_, self.high_ = [], [], []
        self.vocab_: dict[str, list] = {}
        self.freq_: dict[str, dict] = {}
        self.has_unknown_: dict[str, bool] = {}
        self.amt_freq_: dict[int, float] = {}
        self.price_vocab_: list[int] = []

    @staticmethod
    def _cents(amount) -> np.ndarray:
        """Importe como entero en centavos (evita comparar floats)."""
        return np.round(np.asarray(amount, dtype="float64") * 100).astype("int64")

    def fit(self, Xi_train: pd.DataFrame, amount_train, cat_cols):
        n = len(Xi_train)
        self.cols_, self.low_, self.high_ = list(cat_cols), [], []
        for c in self.cols_:
            vc = Xi_train[c].value_counts()
            self.freq_[c] = (vc / n).to_dict()
            frequent = vc[vc >= self.min_freq]
            self.vocab_[c] = list(frequent.index)            # ordenadas por frecuencia desc.
            if len(vc) <= self.onehot_max:
                self.low_.append(c)
                self.has_unknown_[c] = len(frequent) < len(vc)
            else:
                self.high_.append(c)
        vc = pd.Series(self._cents(amount_train)).value_counts()
        self.amt_freq_ = (vc / n).to_dict()
        self.price_vocab_ = list(vc[vc >= self.price_min_freq].index)
        return self

    def transform(self, Xi: pd.DataFrame, amount):
        """Devuelve (X_enc, idx): X_enc con one-hot + frecuencias + amt_freq; idx con índices enteros."""
        parts, idx_cols = {}, {}
        for c in self.low_:
            s = Xi[c]
            for cat in self.vocab_[c]:
                parts[f"{c}__{_safe(cat)}"] = (s == cat).to_numpy(dtype="int8")
            if self.has_unknown_[c]:
                parts[f"{c}__{self.unknown}"] = (~s.isin(self.vocab_[c])).to_numpy(dtype="int8")
        for c in self.high_:
            s = Xi[c]
            parts[f"{c}_freq"] = s.map(self.freq_[c]).fillna(0.0).to_numpy(dtype="float32")
            mapping = {cat: i + 1 for i, cat in enumerate(self.vocab_[c])}
            idx_cols[f"{c}_idx"] = s.map(mapping).fillna(0).to_numpy(dtype="int32")

        cents = pd.Series(self._cents(amount))
        parts["amt_freq"] = cents.map(self.amt_freq_).fillna(0.0).to_numpy(dtype="float32")
        pmap = {v: i + 1 for i, v in enumerate(self.price_vocab_)}
        idx_cols["amt_price_point_idx"] = cents.map(pmap).fillna(0).to_numpy(dtype="int32")

        X_enc = pd.DataFrame(parts, index=Xi.index)
        idx = pd.DataFrame(idx_cols, index=Xi.index)
        assert X_enc.columns.is_unique, "Nombres de columna repetidos tras codificar"
        return X_enc, idx

    def vocab_sizes(self) -> dict[str, int]:
        """Tamaño de vocabulario por índice (incluye el 0 = desconocido), para las capas Embedding."""
        sizes = {f"{c}_idx": len(self.vocab_[c]) + 1 for c in self.high_}
        sizes["amt_price_point_idx"] = len(self.price_vocab_) + 1
        return sizes

    def unknown_rate(self, Xi: pd.DataFrame) -> pd.Series:
        """Por columna: fracción de filas cuya categoría cae en 'desconocido' (rara o nueva)."""
        return pd.Series({c: float((~Xi[c].isin(self.vocab_[c])).mean()) for c in self.cols_})

"""Acción 7. Colas y escalado, ajustados SOLO con train.

    1) Banderas binarias (<= 2 valores distintos en train) y las columnas de cfg no_scale: no se tocan.
    2) Columnas con asimetría > skew_threshold y mínimo >= 0 en train: log1p
       (en dev/test los valores negativos se recortan a 0 antes de aplicarlo).
    3) Estandarización con media y desviación de train (después del log1p).
    Los one-hot, los indicadores de nulo y los índices de embeddings no pasan por aquí.
"""
class NumericTransformer:
    def __init__(self, cfg: dict):
        p = cfg["preprocessing"]
        self.skew_threshold = p["skew_threshold"]
        self.no_scale = set(p.get("no_scale", []))
        self.clip_z = p.get("clip_z")
        self.passthrough_cols_: list[str] = []
        self.log_cols_: list[str] = []
        self.scale_cols_: list[str] = []
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}
        self.skew_: pd.Series | None = None

    def fit(self, X_train: pd.DataFrame):
        cols = list(X_train.columns)
        n_unique = X_train.nunique()
        self.passthrough_cols_ = [c for c in cols if n_unique[c] <= 2 or c in self.no_scale]
        self.scale_cols_ = [c for c in cols if c not in set(self.passthrough_cols_)]

        self.skew_ = X_train[self.scale_cols_].skew()
        mins = X_train[self.scale_cols_].min()
        self.log_cols_ = [c for c in self.scale_cols_
                          if mins[c] >= 0 and self.skew_[c] > self.skew_threshold]

        log_set = set(self.log_cols_)
        for c in self.scale_cols_:                       # una columna a la vez: evita copiar toda la matriz en float64
            v = X_train[c].to_numpy(dtype="float64")
            if c in log_set:
                v = np.log1p(v)
            self.mean_[c] = float(v.mean())
            std = float(v.std())
            self.std_[c] = std if std > 0 else 1.0       # columna constante en train: solo se centra
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        log_set = set(self.log_cols_)
        for c in self.scale_cols_:
            v = X[c].to_numpy(dtype="float64")
            if c in log_set:
                v = np.log1p(np.clip(v, 0, None))
            z = (v - self.mean_[c]) / self.std_[c]
            if self.clip_z is not None:
                z = np.clip(z, -self.clip_z, self.clip_z)
            out[c] = z.astype("float32")
        assert np.isfinite(out.to_numpy(dtype="float64")).all(), "Hay NaN o infinitos tras escalar"
        return out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)


def _corr_matrix(df: pd.DataFrame, chunk: int = 50_000) -> pd.DataFrame:
    """Matriz de correlación de Pearson por bloques (evita copiar toda la matriz en float64)."""
    cols, (n, p) = df.columns, df.shape
    mean = df.mean().to_numpy(dtype="float64")
    G = np.zeros((p, p))
    for i in range(0, n, chunk):
        B = df.iloc[i:i + chunk].to_numpy(dtype="float64") - mean
        G += B.T @ B
    d = np.sqrt(np.diag(G))
    d[d == 0] = np.nan                          # columna constante en train
    return pd.DataFrame(G / np.outer(d, d), index=cols, columns=cols)

def _assoc_with_target(df: pd.DataFrame, y: pd.Series) -> pd.Series:
    """|Spearman| de cada columna con el objetivo (captura relaciones monótonas no lineales)."""
    yc = y.to_numpy(dtype="float64")
    yc = yc - yc.mean()
    ys = np.sqrt((yc ** 2).sum())
    out = {}
    for c in df.columns:
        r = rankdata(df[c].to_numpy())
        r = r - r.mean()
        den = np.sqrt((r ** 2).sum()) * ys
        out[c] = abs(float((r * yc).sum() / den)) if den > 0 else 0.0
    return pd.Series(out)


"""Acción 8. Elimina columnas redundantes. Correlaciones y asociación con el objetivo: SOLO train.

    Recorre las columnas de mayor a menor asociación con el objetivo; una columna se elimina si su
    |r| con alguna columna ya conservada supera el umbral. Así, de cada grupo de columnas casi
    idénticas sobrevive la más asociada con el fraude. Las columnas de corr_protect nunca se eliminan
    y las constantes en train siempre se eliminan.
"""
class RedundancyFilter:

    def __init__(self, cfg: dict):
        p = cfg["preprocessing"]
        self.threshold = p["corr_threshold"]
        self.protect = list(p.get("corr_protect", []))
        self.corr_ = None
        self.assoc_ = None
        self.keep_cols_: list[str] = []
        self.drop_cols_: dict[str, tuple] = {}     # eliminada -> (columna que la reemplaza, |r|)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.corr_ = _corr_matrix(X_train)
        self.assoc_ = _assoc_with_target(X_train, y_train)
        self.keep_cols_, self.drop_cols_ = self.select(self.threshold)
        return self

    def select(self, thr: float):
        cols = list(self.corr_.columns)
        pos = {c: i for i, c in enumerate(cols)}
        const = {c for c in cols if np.isnan(self.corr_.iloc[pos[c], pos[c]])}
        A = np.abs(np.nan_to_num(self.corr_.to_numpy(), nan=0.0))

        prot = [c for c in self.protect if c in pos and c not in const]
        order = [c for c in self.assoc_.sort_values(ascending=False, kind="stable").index
                 if c not in set(prot) and c not in const]
        kept, kept_idx = list(prot), [pos[c] for c in prot]
        dropped = {c: ("constante", 1.0) for c in cols if c in const}
        for c in order:
            if kept_idx:
                row = A[pos[c], kept_idx]
                j = int(row.argmax())
                if row[j] > thr:
                    dropped[c] = (cols[kept_idx[j]], float(row[j]))
                    continue
            kept.append(c)
            kept_idx.append(pos[c])
        kept_set = set(kept)
        return [c for c in cols if c in kept_set], dropped     # conserva el orden original

    def report(self, thresholds) -> pd.DataFrame:
        rows = []
        for t in thresholds:
            keep, drop = self.select(t)
            rows.append({"umbral": t, "eliminadas": len(drop), "se_quedan": len(keep)})
        return pd.DataFrame(rows).set_index("umbral")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.keep_cols_]

    def fit_transform(self, X_train, y_train):
        return self.fit(X_train, y_train).transform(X_train)

class Preprocessor:
    """Acción 10. Orquesta los pasos 2 a 8 del pipeline.

    fit_transform(df_train) ajusta TODO con train (indicadores, imputación, vocabularios, escalado,
    filtro de redundancia). transform(df) aplica lo ajustado a dev/test sin reajustar nada.
    Devuelve un dict con X (tabular), idx (índices para embeddings), y y meta.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.nulls = NullPatternIndicators(min_rows=cfg["preprocessing"]["min_indicator_rows"])
        self.imputer = Imputer(cfg)
        self.encoder = CategoricalEncoder(cfg)
        self.numtf = NumericTransformer(cfg)
        self.rf = RedundancyFilter(cfg)
        self.freq_cols_: list[str] = []
        self.dummy_cols_: list[str] = []
        self.idx_cols_: list[str] = []
        self.feature_names_: list[str] = []
        self.is_fitted_ = False

    def _run(self, df: pd.DataFrame, fit: bool) -> dict:
        amt = self.cfg["columns"]["amount"]
        X, y, meta = prepare_frame(df, self.cfg)                                  # pasos 2 y 3
        I = self.nulls.fit_transform(X) if fit else self.nulls.transform(X)       # paso 4 (antes de imputar)
        Xi = self.imputer.fit_transform(X) if fit else self.imputer.transform(X)  # paso 5
        del X

        if fit:                                                                   # paso 6
            self.encoder.fit(Xi, meta[amt], self.imputer.cat_cols_)
        E, idx = self.encoder.transform(Xi, meta[amt])
        if fit:
            self.freq_cols_ = [c for c in E.columns if c.endswith("_freq")]
            self.dummy_cols_ = [c for c in E.columns if c not in set(self.freq_cols_)]
            self.idx_cols_ = list(idx.columns)

        N = pd.concat([Xi[self.imputer.num_cols_], E[self.freq_cols_]], axis=1)
        del Xi
        Z = self.numtf.fit_transform(N) if fit else self.numtf.transform(N)       # paso 7
        del N
        T = pd.concat([Z, I, E[self.dummy_cols_]], axis=1)
        del Z, I, E

        if fit:                                                                   # paso 8
            self.rf.fit(T, y)
            self.feature_names_ = list(self.rf.keep_cols_)
            self.is_fitted_ = True
        F = self.rf.transform(T)
        del T

        assert list(F.columns) == self.feature_names_, "Columnas distintas a las del ajuste"
        assert not F.isna().any().any(), "Hay NaN en la matriz final"
        assert list(idx.columns) == self.idx_cols_
        return {"X": F, "idx": idx, "y": y, "meta": meta}

    def fit_transform(self, df_train: pd.DataFrame) -> dict:
        return self._run(df_train, fit=True)

    def transform(self, df: pd.DataFrame) -> dict:
        assert self.is_fitted_, "Primero hay que ajustar con train (fit_transform)"
        return self._run(df, fit=False)

    def save(self, path=None):
        mdir = ROOT / self.cfg["paths"]["models_dir"]
        mdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, Path(path) if path else mdir / "preprocessor.joblib")
        meta = {
            "n_features": len(self.feature_names_),
            "feature_names": self.feature_names_,
            "idx_columns": self.idx_cols_,
            "vocab_sizes": self.encoder.vocab_sizes(),
            "null_groups": self.nulls.groups_,
            "log1p_columns": self.numtf.log_cols_,
            "clip_z": self.numtf.clip_z,
            "dropped_by_redundancy": {c: [r, v] for c, (r, v) in self.rf.drop_cols_.items()},
        }
        (mdir / "preprocessor_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(cfg: dict | None = None, path=None) -> "Preprocessor":
        cfg = cfg or load_config()
        p = Path(path) if path else ROOT / cfg["paths"]["models_dir"] / "preprocessor.joblib"
        return joblib.load(p)


def write_splits(outputs: dict, cfg: dict) -> None:
    """Escribe {split}_X / _idx / _y / _meta en data/processed/ (Parquet)."""
    d = ROOT / cfg["paths"]["processed_dir"]
    d.mkdir(parents=True, exist_ok=True)
    target = cfg["columns"]["target"]
    for split, o in outputs.items():
        o["X"].to_parquet(d / f"{split}_X.parquet", index=False)
        o["idx"].to_parquet(d / f"{split}_idx.parquet", index=False)
        o["y"].rename(target).to_frame().to_parquet(d / f"{split}_y.parquet", index=False)
        o["meta"].to_parquet(d / f"{split}_meta.parquet", index=False)


def load_split(cfg: dict, split: str) -> dict:
    """Carga un split ya procesado en una línea: {'X', 'idx', 'y', 'meta'}."""
    d = ROOT / cfg["paths"]["processed_dir"]
    return {k: pd.read_parquet(d / f"{split}_{k}.parquet") for k in ("X", "idx", "y", "meta")}