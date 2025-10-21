#==============================================================
# Script : train_rf.py
# Description : Version initiale d’entraînement du modèle Random Forest
# Auteur : Corentin CARTALLIER
# Statut : Déprécié / Non recommandé pour usage réel
#==============================================================
#
# DESCRIPTION :
# Ce script constitue la première version de l’entraînement du modèle Random Forest
# sur le dataset CICIDS2017 (MachineLearningCSV). Il réalise une fusion complète
# des fichiers de capture réseau et entraîne un classifieur RandomForestClassifier
# sur un sous-échantillon des données.
#
# LIMITES TECHNIQUES ET PROBLÈMES RENCONTRÉS :
# - Les colonnes contenant des valeurs infinies (Flow Bytes/s, Flow Packets/s, etc.)
#   ne sont pas correctement gérées, ce qui provoque des erreurs de type :
#     ValueError: Input X contains infinity or a value too large for dtype('float32')
#
# - Le nettoyage des NaN est simpliste et ne prend pas en compte les variations
#   entre colonnes (remplissage global au lieu de par colonne).
#
# - Aucune normalisation (StandardScaler) n’est appliquée, ce qui crée de fortes
#   disparités d’échelle entre features et perturbe la convergence du modèle.
#
# - Le dataset complet (plus de 2.8 millions de lignes) est chargé en mémoire,
#   ce qui rend l’entraînement très long et potentiellement instable selon la RAM.
#
# - L’échantillonnage "FAST_MODE" (10%) fausse les proportions d’attaques réelles
#   et peut entraîner un surapprentissage artificiel (accuracy de 99–100%).
#
# - Le modèle est entraîné et évalué sur les mêmes jours, donc aucune vraie
#   séparation temporelle. Cela biaise les résultats car le modèle "revoit"
#   des patterns déjà connus.
#
# - Le code ne gère pas les colonnes non numériques, ce qui peut provoquer
#   des erreurs silencieuses si des colonnes non nettoyées subsistent.
#
# EN CONSÉQUENCE :
# → Ce script donne des résultats *trop parfaits* (accuracy = 1.00) qui sont
#   trompeurs et non représentatifs des performances réelles.
# → Il est non adapté à un usage de production ou d’évaluation sérieuse.
#
# Ce fichier est conservé uniquement à titre historique pour suivre
# l’évolution du projet et des versions successives de l’algorithme.
#==============================================================


import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


# === Chemins ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "dataset_prepared.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "model_rf.joblib")

# === Chargement des données ===
print(" Chargement du dataset...")
df = pd.read_csv(DATA_FILE)
print(f" Dataset chargé : {df.shape}")

# === Vérification des colonnes suspectes ===
print("\n🔎 Vérification des colonnes suspectes contenant 'label' ou similaires...")
suspicious_cols = [c for c in df.columns if "label" in c.lower()]
if suspicious_cols:
    print(f" Colonnes suspectes détectées : {suspicious_cols}")
else:
    print(" Aucune colonne 'label' détectée dans le dataset.")

# === Vérification des corrélations trop fortes ===
if "Attack_Flag" in df.columns:
    print("\n Vérification des corrélations fortes avec 'Attack_Flag'...")
    corr = df.corr(numeric_only=True)["Attack_Flag"].sort_values(ascending=False)
    print(corr.head(10))
    high_corr = corr[abs(corr) > 0.95].drop("Attack_Flag", errors="ignore")
    if not high_corr.empty:
        print(" Colonnes très corrélées à la cible (risque de fuite) :")
        print(high_corr)
    else:
        print(" Aucune corrélation suspecte (>0.95) détectée.")
else:
    print("Colonne 'Attack_Flag' absente, corrélation non vérifiée.")

# === Séparation features / label ===
X = df.drop(columns=["Label", "Attack_Flag"], errors="ignore")
y = df["Attack_Flag"]

# === Vérification structure des features ===
print("\n Exemple des dernières colonnes :")
print(X.columns[-10:].tolist())

# === Mode rapide (optionnel) ===
FAST_MODE = True  # Mettre à True pour un entraînement rapide
if FAST_MODE:
    print("\n Mode rapide activé (échantillon 10%)")
    X, _, y, _ = train_test_split(X, y, train_size=0.1, stratify=y, random_state=42)

# === Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f" Jeu d'entraînement : {X_train.shape}, test : {X_test.shape}")

# === Modèle Random Forest ===
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nEntraînement du modèle (avec barre de progression)...")
with tqdm_joblib(tqdm(desc=" Entraînement RandomForest", total=rf.n_estimators)) as progress_bar:
    rf.fit(X_train, y_train)
print("\nEntraînement terminé !\n")

# === Évaluation ===
print("Évaluation du modèle :")
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC :", roc_auc_score(y_test, y_prob))

# === Matrice de confusion ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion - Random Forest")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.tight_layout()
plt.show()

# === Sauvegarde du modèle ===
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(rf, MODEL_FILE)
print(f"Modèle sauvegardé sous : {MODEL_FILE}")

# === Test automatique ===
assert os.path.exists(MODEL_FILE), "Le modèle n’a pas été sauvegardé."
print("Test automatique passé : modèle bien enregistré.")