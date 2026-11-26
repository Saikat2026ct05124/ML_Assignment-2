"""Loading, cleaning, splitting and the sklearn preprocessing transformer."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def load_raw() -> pd.DataFrame:
    return pd.read_csv(C.DATA_PATH)  # the string 'NULL' is parsed as NaN


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, leakage columns; fill structural missing values."""
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {n_dup} duplicate rows -> {df.shape}")
    df = df.drop(columns=C.LEAKAGE_COLS + C.DROP_COLS)
    df["agent"] = df["agent"].fillna(-1).astype(int).astype(str).replace("-1", "no_agent")
    df["country"] = df["country"].fillna("Unknown")
    return df  # 'children' NaN is imputed inside the pipeline (train median)


def split(df: pd.DataFrame):
    X, y = df.drop(columns=[C.TARGET]), df[C.TARGET]
    return train_test_split(X, y, test_size=C.TEST_SIZE, random_state=C.SEED, stratify=y)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute+scale numerics, one-hot categoricals. Fitted on train only (inside Pipeline)."""
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False), cat_cols),
    ])
