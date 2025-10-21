# Script : preprocess_data.py
# Description : Nettoyage, encodage et normalisation du dataset CICIDS2017
# Auteur : Corentin CARTALLIER

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np

#  Définition des chemins 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "MachineLearningCVE")
OUTPUT_FILE = os.path.join(DATA_DIR, "dataset_prepared.csv")

#Chargement et fusion des fichiers
def load_data():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(RAW_DIR, f))
        df.columns = df.columns.str.strip()
        dfs.append(df)
    df_full = pd.concat(dfs, ignore_index=True)
    print(f"Données fusionnées : {df_full.shape}")
    return df_full

# Nettoyage des labels et suppression des colonnes inutiles

def clean_data(df):
    # Nettoyage du label
    df["Label"] = df["Label"].astype(str).str.strip()
    df["Label"] = df["Label"].str.replace("�", "-", regex=False)
    df["Label"] = df["Label"].replace({
        "Web Attack - Brute Force": "Web Attack",
        "Web Attack - XSS": "Web Attack",
        "Web Attack - Sql Injection": "Web Attack"
    })

    # Suppression des colonnes non pertinentes ou inutilisables
    drop_cols = ["Flow ID", " Source IP", " Destination IP", " Timestamp"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Suppression des valeurs infinies ou NaN
    df = df.replace([float('inf'), float('-inf')], pd.NA)
    df = df.dropna()

    print(f" Après nettoyage : {df.shape}")
    return df

# Conversion du label en binaire
def encode_label(df):
    df["Attack_Flag"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    print(df["Attack_Flag"].value_counts())
    return df

# Normalisation des variables numériques
def normalize_data(df):
    # Exclure Attack_Flag du scaling
    features = [col for col in df.select_dtypes(include=["float64", "int64"]).columns if col != "Attack_Flag"]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    print(f"Normalisation appliquée à {len(features)} features (Attack_Flag exclu).")
    return df

def save_prepared_data(df, output_file=OUTPUT_FILE, chunk_size=50000):
    print(f"Sauvegarde progressive vers : {output_file}")
    total_rows = len(df)
    chunks = np.arange(0, total_rows, chunk_size)

    # Supprime l'ancien fichier s'il existe
    if os.path.exists(output_file):
        os.remove(output_file)

    # Écriture par morceaux
    with tqdm(total=len(chunks), desc="Écriture CSV", unit="chunk") as pbar:
        for i in chunks:
            chunk = df.iloc[i:i + chunk_size]
            header = i == 0  # écrire l'entête seulement au premier chunk
            chunk.to_csv(output_file, mode='a', header=header, index=False)
            pbar.update(1)

    print(f"\nÉcriture terminée ({total_rows:,} lignes sauvegardées).")

# Programme principal
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = encode_label(df)
    df = normalize_data(df)
    save_prepared_data(df)
