# Script: explore_dataset.py
# Description: Exploration statistique et graphique du dataset CICIDS2017
# Auteur: Corentin CARTALLIER

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Définition des chemins du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "MachineLearningCSV")

def load_and_sample_data(sample_size_per_file=None):
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    dfs = []
    print(f"{len(files)} fichiers trouvés : {files}\n")

    for f in files:
        path = os.path.join(DATA_DIR, f)
        if sample_size_per_file:
            print(f"Lecture partielle de {f} ({sample_size_per_file} lignes)")
            df = pd.read_csv(path, nrows=sample_size_per_file)
        else:
            print(f"Lecture complète de {f}")
            df = pd.read_csv(path)

        df.columns = df.columns.str.strip()
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Données fusionnées : {full_df.shape[0]} lignes, {full_df.shape[1]} colonnes")
    return full_df
#Analyse basique des colonnes et labels
def summarize_dataset(df):
    print("\n Aperçu des données :")
    print(df.head(3))

    print("\n Types de colonnes :")
    print(df.dtypes.value_counts())

    print("\n Distribution des labels :")
    print(df["Label"].value_counts())

#Visualisations de la distribution des labels
def plot_label_distribution(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 5))
    order = df["Label"].value_counts().index
    sns.countplot(x="Label", data=df, order=order, palette="viridis")
    plt.title("Distribution des types de trafic (CICIDS2017)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Regroupement binaire : Benign vs Attack
    df["Attack_Flag"] = df["Label"].apply(lambda x: "Attack" if x != "BENIGN" else "Benign")
    plt.figure(figsize=(5, 5))
    df["Attack_Flag"].value_counts().plot.pie(autopct="%1.1f%%", colors=["#4CAF50", "#F44336"])
    plt.title("Répartition globale : Benign vs Attack")
    plt.ylabel("")
    plt.show()

# Programme principal
if __name__ == "__main__":
    df = load_and_sample_data()  
    plot_label_distribution(df)
