"""
Automasi Preprocessing Dataset Telco Customer Churn.

Script ini mengkonversi proses eksperimen dari notebook ke fungsi Python
yang dapat dijalankan secara otomatis untuk menghasilkan data yang siap dilatih.

Tahapan:
1. Data Loading   - Memuat dataset mentah dari folder namadataset_raw
2. Data Cleaning  - Handle missing values, drop kolom tidak relevan
3. Feature Engineering - Encoding kategorikal, transformasi fitur
4. Scaling         - Standarisasi fitur numerik
5. Save            - Simpan dataset siap latih ke namadataset_preprocessing

Author: Muhamad Hafizh Albar
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(raw_data_path: str) -> pd.DataFrame:
    """Tahap 1: Memuat dataset mentah."""
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {raw_data_path}. "
            "Pastikan file CSV sudah ada di folder namadataset_raw."
        )

    df = pd.read_csv(raw_data_path)
    print(f"[LOAD] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"[LOAD] Kolom: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Tahap 2: Membersihkan data - handle missing values & kolom tidak relevan."""
    # Drop kolom customerID karena tidak relevan untuk pemodelan
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("[CLEAN] Kolom 'customerID' dihapus")

    # TotalCharges memiliki nilai string kosong (" ") yang perlu di-handle
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        missing_count = df["TotalCharges"].isna().sum()
        if missing_count > 0:
            median_val = df["TotalCharges"].median()
            df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
            print(
                f"[CLEAN] {missing_count} missing values di 'TotalCharges' "
                f"diisi dengan median ({median_val:.2f})"
            )

    # Cek dan handle missing values lainnya
    total_missing = df.isna().sum().sum()
    if total_missing > 0:
        print(f"[CLEAN] Total missing values tersisa: {total_missing}")
        # Isi numerik dengan median, kategorikal dengan modus
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        print("[CLEAN] Semua missing values telah ditangani")

    # Drop baris duplikat
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"[CLEAN] {dup_count} baris duplikat dihapus")
    else:
        print("[CLEAN] Tidak ada baris duplikat")

    print(f"[CLEAN] Dimensi setelah cleaning: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tahap 3: Encoding fitur kategorikal dan target variabel."""
    # Encode target variable: Churn (Yes/No -> 1/0)
    if "Churn" in df.columns:
        le = LabelEncoder()
        df["Churn"] = le.fit_transform(df["Churn"])
        print(f"[ENCODE] Target 'Churn' diubah ke numerik: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Identifikasi kolom kategorikal (object type)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if cat_cols:
        print(f"[ENCODE] Kolom kategorikal yang akan di-encode: {cat_cols}")
        # One-Hot Encoding untuk fitur kategorikal
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
        print(f"[ENCODE] Setelah One-Hot Encoding: {df.shape[1]} kolom")

    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tahap 4: Standarisasi fitur numerik."""
    # Kolom numerik yang perlu di-scale (kecuali target dan binary)
    target_col = "Churn"
    numeric_cols = []

    for col in df.columns:
        if col == target_col:
            continue
        # Hanya scale kolom yang bukan binary (0/1)
        unique_vals = df[col].nunique()
        if unique_vals > 2 and df[col].dtype in ["float64", "int64"]:
            numeric_cols.append(col)

    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print(f"[SCALE] Kolom yang di-standarisasi: {numeric_cols}")

    return df


def preprocess(raw_data_path: str = None, output_path: str = None) -> pd.DataFrame:
    """
    Fungsi utama preprocessing yang menjalankan seluruh pipeline.
    
    Mengembalikan DataFrame yang siap untuk dilatih model ML.
    
    Parameters:
        raw_data_path: Path ke file CSV mentah
        output_path: Path untuk menyimpan hasil preprocessing
    
    Returns:
        pd.DataFrame: Data yang sudah dipreprocessing dan siap dilatih
    """
    # Default paths berdasarkan struktur folder kriteria 1
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if raw_data_path is None:
        raw_data_path = os.path.join(
            base_dir, "..", "namadataset_raw", "Dataset-Telco-Customer-Churn.csv"
        )
    
    if output_path is None:
        output_path = os.path.join(
            base_dir, "namadataset_preprocessing", "clean_data.csv"
        )

    print("=" * 60)
    print("PIPELINE PREPROCESSING OTOMATIS")
    print("Dataset: Telco Customer Churn")
    print("=" * 60)

    # Tahap 1: Data Loading
    df = load_data(raw_data_path)

    # Tahap 2: Data Cleaning
    df = clean_data(df)

    # Tahap 3: Feature Engineering (Encoding)
    df = encode_features(df)

    # Tahap 4: Scaling
    df = scale_features(df)

    # Tahap 5: Simpan hasil preprocessing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[SAVE] Dataset preprocessed disimpan ke: {output_path}")
    print(f"[SAVE] Dimensi akhir: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"[SAVE] Kolom target: Churn")
    print(f"[SAVE] Distribusi target:\n{df['Churn'].value_counts().to_string()}")

    # Juga simpan ke root-level namadataset_preprocessing
    root_output = os.path.join(base_dir, "..", "namadataset_preprocessing", "clean_data.csv")
    os.makedirs(os.path.dirname(root_output), exist_ok=True)
    df.to_csv(root_output, index=False)
    print(f"[SAVE] Salinan juga disimpan ke: {root_output}")

    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI!")
    print("=" * 60)

    return df


def main() -> None:
    """Entry point untuk menjalankan preprocessing secara otomatis."""
    preprocess()


if __name__ == "__main__":
    main()
