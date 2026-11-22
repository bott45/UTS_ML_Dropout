"""Praproses untuk dataset Student Dropout (UCI 697).

Semua langkah ada di dalam ColumnTransformer/Pipeline, jadi hanya di-fit
pada data latih (tidak ada kebocoran data saat cross-validation).
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "dropout"

# Kolom berisi kode angka (kategori, bukan angka berurutan) -> one-hot
KATEGORIK = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]

# Kolom biner 0/1 -> dibiarkan apa adanya
BINER = [
    "Daytime/evening attendance", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder",
    "International",
]


def bersihkan_nama_kolom(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus spasi/tab di ujung nama kolom (mis. 'Daytime/evening attendance\\t')."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def muat_data(path_train="../data/processed/train.csv"):
    """Baca train.csv, kembalikan X (fitur) dan y (label dropout)."""
    df = bersihkan_nama_kolom(pd.read_csv(path_train))
    return df.drop(columns=[TARGET]), df[TARGET]


def kolom_akademik(X: pd.DataFrame):
    """Kolom kinerja akademik semester 1 dan 2 (hanya ada setelah kuliah berjalan)."""
    return [c for c in X.columns if c.startswith("Curricular units")]


def pilih_skenario(X: pd.DataFrame, dengan_akademik: bool) -> pd.DataFrame:
    """Skenario 'tanpa' membuang kolom akademik semester (peringatan dini)."""
    if dengan_akademik:
        return X
    return X.drop(columns=kolom_akademik(X))


def buat_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Susun praproses dari kolom yang ada di X."""
    kat = [c for c in KATEGORIK if c in X.columns]
    biner = [c for c in BINER if c in X.columns]
    numerik = [c for c in X.columns if c not in kat + biner]
    return ColumnTransformer(
        transformers=[
            ("kat", OneHotEncoder(handle_unknown="ignore"), kat),
            ("num", StandardScaler(), numerik),
            ("biner", "passthrough", biner),
        ]
    )


def buat_pipeline(model, X: pd.DataFrame) -> Pipeline:
    """Praproses + model dalam satu Pipeline."""
    return Pipeline([("praproses", buat_preprocessor(X)), ("model", model)])