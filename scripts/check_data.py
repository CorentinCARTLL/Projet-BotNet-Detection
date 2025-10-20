# Script: check_data.py
# Description: Ce fichier a pour but de vérifier la présence et l'intégrité des fichiers de données CSV (la base de données CICIDS2017)
# Auteur: Corentin CARTALLIER

import os
import pandas as pd

# Déterminer le chemin absolu du dossier racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "MachineLearningCSV")

def verify_data():
    print(f"Lecture depuis : {DATA_DIR}")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    print(f"\nFichiers CSV détectés : {len(files)}")
    for f in files[:5]:  # Afficher les 5 premiers fichiers
        print(" -", f)

    sample = os.path.join(DATA_DIR, files[0]) # Prendre le premier fichier comme échantillon
    df = pd.read_csv(sample, nrows=50000) # Lire un échantillon de 50 000 lignes
    print("\nColonnes disponibles :")
    print(df.columns.tolist()) # Afficher la liste des colonnes
    print("\nLabels uniques :", df[" Label"].unique()[:5]) # Afficher les 5 premiers labels uniques


#Execution du main
if __name__ == "__main__":
    verify_data()
