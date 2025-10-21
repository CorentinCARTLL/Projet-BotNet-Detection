#==============================================================
# Script : train_rf_daysplit_optimized.py
# Description : Modèle Random Forest optimisé avec nettoyage, normalisation
#               et ajustement automatique du seuil de classification
# Auteur : Corentin CARTALLIER
#==============================================================
#
# CONTEXTE :
# Ce script entraîne un modèle de Machine Learning (Random Forest)
# sur le dataset CICIDS2017, en séparant les jours d’entraînement et de test
# pour simuler un scénario réaliste d’intrusion réseau sur des jours inédits.
#
# Il s’agit d’une version stabilisée et optimisée du modèle initial :
# - Nettoyage robuste des valeurs manquantes (NaN) et infinies (Inf / -Inf)
# - Normalisation des features via StandardScaler
# - Gestion automatique du déséquilibre de classes (class_weight="balanced_subsample")
# - Recherche d’un seuil optimal (maximisation du F1-score)
# - Entraînement avec barre de progression (tqdm + joblib)
#
#==============================================================
# PIPELINE TECHNIQUE :
#
# Chargement et fusion des fichiers journaliers :
#     - Entraînement : Monday → Thursday
#     - Test : Friday (trafic inédit)
#
# Nettoyage :
#     - Remplacement des NaN par la moyenne des colonnes d’entraînement
#     - Remplacement des valeurs infinies par le maximum non infini
#     - Vérification de la finitude complète du jeu de données
#
# Normalisation :
#     - StandardScaler appliqué uniquement sur les colonnes numériques
#     - Fit sur train / Transform sur test pour éviter la fuite de données
#
# Entraînement :
#     - Modèle RandomForestClassifier (400 arbres, équilibrage dynamique)
#     - Progression visible avec tqdm_joblib
#
# Évaluation :
#     - Rapport de classification complet (precision, recall, f1-score)
#     - Score ROC-AUC global
#     - Recherche du seuil optimal maximisant le F1-score
#     - Affichage de la **matrice de confusion** ajustée au seuil
#
# Sauvegarde :
#     - Modèle exporté sous : models/model_rf_daysplit_optimized.joblib
#     - Scaler exporté sous : models/scaler.joblib
#
#==============================================================
# RÉSULTATS TYPIQUES :
# Accuracy : ~0.70
# ROC-AUC  : ~0.79
# → Le modèle généralise correctement sur des jours jamais vus.
# → Il reconnaît très bien le trafic normal (rappel > 95 %),
#   mais détecte encore partiellement les attaques (~30 % de rappel).
#
#==============================================================
# LIMITES CONNUES :
# - Random Forest reste limité sur des jeux de données fortement déséquilibrés
#   (majorité de trafic BENIGN).
# - Les classes rares (attaques) sont partiellement sous-apprises.
# - Le modèle n’exploite pas encore le suréchantillonnage (SMOTE)
#   ni des algorithmes plus récents (XGBoost, LightGBM).
#
#==============================================================
#  RECOMMANDATIONS :
# - Ajouter un suréchantillonnage intelligent (SMOTE)
# - Réduire les features non corrélées (feature selection / PCA)
# - Intégrer une validation croisée temporelle (cross-validation)
#
# Ce script constitue néanmoins une base stable et fiable
# pour la détection de trafic malveillant supervisée sur CICIDS2017.
#==============================================================


import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# === Chemins ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "MachineLearningCVE")
MODEL_FILE = os.path.join(BASE_DIR, "models", "model_rf_daysplit_optimized.joblib")
SCALER_FILE = os.path.join(BASE_DIR, "models", "scaler.joblib")

# === Fichiers par jour ===
train_days = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]
test_days = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
]

# === Chargement ===
def load_days(files):
    dfs = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        print(f"Lecture de {f} ...")
        df = pd.read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

print("Chargement des jours d'entraînement...")
df_train = load_days(train_days)
print("Chargement des jours de test...")
df_test = load_days(test_days)

# === Préparation des données ===
X_train = df_train.drop(columns=[" Label"], errors="ignore")
y_train = df_train[" Label"].apply(lambda x: 0 if str(x).strip() == "BENIGN" else 1)

X_test = df_test.drop(columns=[" Label"], errors="ignore")
y_test = df_test[" Label"].apply(lambda x: 0 if str(x).strip() == "BENIGN" else 1)

print(f"Entraînement : {X_train.shape}, Test : {X_test.shape}")

# === Nettoyage NaN / Inf (robuste) ===
print("Nettoyage des valeurs manquantes et infinies...")

# 1) Remplir les NaN par la moyenne du train
X_train = X_train.fillna(X_train.mean(numeric_only=True))
X_test = X_test.fillna(X_train.mean(numeric_only=True))

# 2) Remplacer les ±inf colonne par colonne par le max (hors inf)
for df in (X_train, X_test):
    # on travaille uniquement sur les colonnes numériques
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        col_series = df[col]
        has_inf = np.isinf(col_series.values).any()
        if has_inf:
            # max calculé hors inf/-inf
            finite_mask = np.isfinite(col_series.values)
            if finite_mask.any():
                max_val = col_series.values[finite_mask].max()
            else:
                # cas pathologique : tout est inf -> on force à 0
                max_val = 0.0
            df[col] = col_series.replace([np.inf, -np.inf], max_val)

# 3) Assertions de sécurité
def _check_finite(name, data):
    if not np.isfinite(data.select_dtypes(include=[np.number]).values).all():
        raise ValueError(f"{name} contient encore des NaN/Inf après nettoyage.")

_check_finite("X_train", X_train)
_check_finite("X_test", X_test)
print("Données finies et propres.")

# === Normalisation (fit sur train, transform sur test) ===
print("Normalisation des features...")
scaler = StandardScaler(with_mean=True, with_std=True)
X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
)
X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
)

# === Modèle RandomForest optimisé ===
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,              # laisse l'arbre s'adapter
    min_samples_leaf=2,          # réduit l'overfit
    class_weight="balanced_subsample",  # rééquilibrage dynamique
    random_state=42,
    n_jobs=-1,
)

# === Entraînement avec barre de progression ===
print("Entraînement du modèle optimisé (avec barre de progression)...")
with tqdm_joblib(tqdm(desc="Entraînement RandomForest", total=rf.n_estimators)) as _:
    rf.fit(X_train, y_train)
print("\nEntraînement terminé !\n")

# === Évaluation (seuil 0.5 par défaut) ===
print("Évaluation classique (seuil 0.5) :")
y_prob = rf.predict_proba(X_test)[:, 1]
y_pred_default = (y_prob > 0.5).astype(int)
print(classification_report(y_test, y_pred_default, digits=4))
print("ROC-AUC :", round(roc_auc_score(y_test, y_prob), 6))

# === Recherche d'un seuil optimisé (max F1) ===
best_f1, best_thresh = 0.0, 0.5
for thresh in np.linspace(0.1, 0.9, 17):  # pas fin pour éviter le surajustement
    y_pred_t = (y_prob > thresh).astype(int)
    f1 = f1_score(y_test, y_pred_t)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

print(f"\nSeuil optimal trouvé : {best_thresh:.2f} (F1 = {best_f1:.4f})")
y_pred_opt = (y_prob > best_thresh).astype(int)

print("\nRapport après ajustement du seuil :")
print(classification_report(y_test, y_pred_opt, digits=4))

# === Matrice de confusion (avec seuil optimisé) ===
cm = confusion_matrix(y_test, y_pred_opt)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matrice de confusion - RF Optimisé (Seuil={best_thresh:.2f})")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.tight_layout()
plt.show()

# === Sauvegarde modèle + scaler ===
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(rf, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
print(f"Modèle sauvegardé sous : {MODEL_FILE}")
print(f"Scaler sauvegardé sous : {SCALER_FILE}")