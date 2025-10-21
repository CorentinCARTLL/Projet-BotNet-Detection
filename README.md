# Projet BotNet Detection — Cyber Threat Intelligence via Machine Learning

##  Description

Le projet **BotNet Detection** est une solution de **détection intelligente d’activités malveillantes dans le trafic réseau**.
Basé sur le jeu de données **CICIDS2017**, ce projet applique des modèles d’apprentissage supervisé (Random Forest et XGBoost) pour **identifier automatiquement les attaques réseau** telles que :

*  **DDoS** (Distributed Denial of Service)
*  **Port Scan**
*  **Infiltration & Web Attacks**
*  **Botnets et trafic anormal**

Ce projet combine **ingénierie de la donnée**, **cybersécurité** et **intelligence artificielle**.
L’objectif est clair : **entraîner un modèle capable de détecter des comportements anormaux même sur des jours jamais vus auparavant** — une approche inspirée des systèmes **IDS/IPS (Intrusion Detection Systems)** modernes.

##  Architecture du projet

Projet-BotNet-Detection/
  * data/                        # Jeux de données (non inclus ici)
    * MachineLearningCSV/         # Dossier à créer et remplir manuellement
  * models/                      # Modèles entraînés (.joblib)
  * scripts/
    * check_data.py               # Vérifie la présence et l’intégrité des fichiers CSV
    * preprocess_data.py          # Nettoyage et préparation du dataset
    * explore_dataset.py          # Exploration et visualisation des données
    * train_rf_daysplit_optimized.py   # Random Forest optimisé
  * requirements.txt             # Dépendances Python
  * README.md

## Installation et configuration

### 1. Cloner le dépôt

git clone [https://github.com/CorentinCARTLL/Projet-BotNet-Detection.git](https://github.com/CorentinCARTLL/Projet-BotNet-Detection.git)
cd Projet-BotNet-Detection

### 2. Créer et activer l’environnement virtuel

python -m venv venv
Sous Windows : venv\Scripts\activate
Sous Linux / Mac : source venv/bin/activate

### 3. Installer les dépendances nécessaires

pip install --upgrade pip
pip install -r requirements.txt

### 4. Télécharger le dataset CICIDS2017

Les fichiers ne sont **pas inclus** dans le dépôt GitHub (taille trop importante).

Télécharge-les depuis le site officiel :
[https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)

Ensuite :

* Extrais le dossier `MachineLearningCSV` dans `Projet-BotNet-Detection/data/`
* Tu devrais obtenir :

  * data/

    * MachineLearningCSV/

      * Monday-WorkingHours.pcap_ISCX.csv
      * Tuesday-WorkingHours.pcap_ISCX.csv
      * ...
      * Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

## Pipeline d’apprentissage automatique

1. **Vérification des données** (scripts/check_data.py)
   S’assure que tous les fichiers sont présents et correctement formatés.

2. **Préparation / Nettoyage** (scripts/preprocess_data.py)

   * Suppression des valeurs NaN et infinies
   * Normalisation (StandardScaler)
   * Encodage des labels

3. **Exploration** (scripts/explore_dataset.py)
   Analyse visuelle : distribution des classes, corrélations, histogrammes.

4. **Entraînement ML**

   * train_rf_daysplit_optimized.py : Random Forest optimisé
   * train_xgboost_daysplit.py : XGBoost pour de meilleures performances

5. **Évaluation et inférence** (scripts/infer.py)
   Permet de charger un modèle et de détecter des attaques sur de nouvelles données.

## Résultats et performances

Les modèles ont été évalués sur le jour de test "Friday" (données jamais vues).
Cette séparation temporelle simule un scénario réaliste de **cybersécurité en production**.

| Modèle                      | Accuracy | Recall (Attack) | F1-Score | ROC-AUC | Commentaire                                 |
| ----------------------------| -------- | --------------- | -------- | ------- | --------------------------------------------|
|  **Random Forest Optimisé** | 0.70     | 0.28            | 0.44     | 0.79    | Bon modèle de base, stable et interprétable |
|  **Baseline (RF Simple)**   | 0.61     | 0.06            | 0.11     | 0.79    | Version d’essai, à ne plus utiliser         |

## Interprétation

* Ils apprennent à différencier efficacement le trafic **BENIGN** des **attaques**.
* L’approche par **séparation de jours (day-split)** garantit une **robustesse temporelle**.
* La Random Forest offre une **forte stabilité**, tandis que XGBoost apporte une **généralisation supérieure**.

>  En résumé : le système reproduit le fonctionnement d’un **IDS (Intrusion Detection System)** basé sur l’apprentissage supervisé.

##  Technologies utilisées

| Domaine                     | Bibliothèques                   |
| --------------------------- | ------------------------------- |
| **Data Processing**         | Pandas, Numpy, Joblib           |
| **Machine Learning**        | Scikit-Learn, XGBoost           |
| **Visualization**           | Matplotlib, Seaborn             |
| **Optimisation**            | TQDM, tqdm-joblib               |
| **Cybersécurité appliquée** | Dataset CICIDS2017 (UNB Canada) |

##  Enjeux cybersécurité

Ce projet illustre le rôle fondamental de l’IA dans la **cyberdéfense moderne** :

* Détection proactive des anomalies réseau.
* Réduction de la charge humaine sur l’analyse des logs.
* Identification des attaques émergentes à partir de signatures comportementales.
* Base pour la création d’un **système IDS intelligent auto-apprenant**.

## Améliorations futures

| Axe                            | Objectif                                                        |
| ------------------------------ | --------------------------------------------------------------- |
|  **SMOTE / ADASYN**          | Rééquilibrer les classes pour mieux détecter les attaques rares |
|  **Feature Selection / PCA** | Réduire les dimensions pour optimiser la vitesse                |
|  **LightGBM / CatBoost**     | Tester d’autres algorithmes de boosting                         |
|  **Déploiement Streamlit**   | Créer une interface interactive de visualisation des alertes    |
|  **MLOps / Logging**         | Automatiser le suivi et la mise à jour du modèle                |

##  Auteur

**Corentin CARTALLIER**
Étudiant ingénieur en informatique et cybersécurité – CESI Toulouse
Passionné par l’intelligence artificielle appliquée à la cybersécurité et la détection d’anomalies réseau.

[LinkedIn](https://www.linkedin.com/in/corentin-cartallier-71a56035a/)
[GitHub](https://github.com/CorentinCARTLL)

##  Licence

Projet open-source sous licence **MIT** — libre d’utilisation et de modification à des fins éducatives, expérimentales et de recherche.

**Si vous trouvez ce projet utile, pensez à lui attribuer une étoile sur GitHub !**
