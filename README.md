#  TourPedia Paris — Big Data & ML Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ascUC2Vd1ykM4DLXIM82gNPz17vGhQM6?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org)

> Analyse complète du dataset TourPedia Paris : **56 361 lieux** (restaurants, hôtels, attractions, POIs)
> et **86 000+ avis** collectés depuis Foursquare, Google Places et Facebook.

---

## Présentation

Ce projet démontre un pipeline **end-to-end** de Data Science appliqué à des données réelles de tourisme
parisien : chargement, nettoyage, exploration, visualisation géospatiale, clustering et modélisation
prédictive — le tout dans un notebook Google Colab exécutable en un clic.

### Objectifs
| # | Objectif | Technique |
|---|---|---|
| 1 | Explorer la structure du dataset | EDA, statistiques descriptives |
| 2 | Nettoyer et valider les données | Pandas, filtrage géospatial |
| 3 | Cartographier les lieux de Paris | Folium HeatMap, Plotly MapBox |
| 4 | Identifier les zones touristiques | K-Means clustering |
| 5 | Prédire le score d'un lieu | Random Forest, Gradient Boosting, Ridge |
| 6 | Comparer les modèles | Cross-validation, MAE, RMSE, R² |

---

## Résultats clés

| Métrique | Valeur |
|---|---|
| Lieux analysés | **56 361** |
| Reviews totales | **86 000+** |
| Sources | Foursquare · Google Places · Facebook |
| Meilleur modèle | Gradient Boosting — R² ≈ **0.72** |
| Zones touristiques détectées | **8 clusters** K-Means |
| Polarité moyenne des avis | **5.6 / 10** |

---

##  Structure du projet

```
tourpedia-paris-analysis/
│
├── TourPedia_Paris_Analysis.ipynb   ← Notebook principal (Google Colab)
├── requirements.txt                  ← Dépendances Python
├── README.md                         ← Ce fichier
│
└── outputs/                          ← Générés à l'exécution
    ├── map_heatmap.html              ← Carte de chaleur interactive
    └── map_top200.html               ← Top 200 lieux interactif
```

---

## Architecture du pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        tourPedia_paris.json                         │
│            (JSONL · 61 Mo · 56 361 enregistrements)                │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     1. CHARGEMENT          │
                    │  load_jsonl() → list[dict] │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     2. TRANSFORMATION      │
                    │  flatten_record() → df     │
                    │  Extraction GPS, reviews   │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     3. NETTOYAGE           │
                    │  Suppression NaN GPS       │
                    │  Filtrage bbox Paris       │
                    │  Déduplication             │
                    └──────┬──────────┬──────────┘
                           │          │
           ┌───────────────▼──┐   ┌───▼───────────────────┐
           │   4. EDA &       │   │  5. MACHINE LEARNING   │
           │   VISUALISATION  │   │                        │
           │                  │   │  ┌─────────────────┐   │
           │  • Distributions │   │  │ K-Means         │   │
           │  • Top lieux     │   │  │ Clustering      │   │
           │  • Folium maps   │   │  │ (géospatial)    │   │
           │  • Plotly charts │   │  └────────┬────────┘   │
           └──────────────────┘   │           │             │
                                  │  ┌────────▼────────┐   │
                                  │  │ Régression      │   │
                                  │  │ Ridge / RF /    │   │
                                  │  │ Gradient Boost  │   │
                                  │  └────────┬────────┘   │
                                  └───────────┼────────────┘
                                              │
                              ┌───────────────▼────────────────┐
                              │     6. ÉVALUATION & DASHBOARD  │
                              │  MAE · RMSE · R² · CV 5-fold   │
                              │  Feature importance            │
                              │  Plotly dashboard interactif   │
                              └────────────────────────────────┘
```

---

## Lancer le projet

### Option 1 — Google Colab (recommandé, zéro installation)
1. Cliquer sur le badge **Open in Colab** en haut du README
2. Dans Colab : `Fichier → Sauvegarder une copie dans Drive`
3. Uploader `tourPedia_paris.json` via la cellule d'upload
4. `Exécution → Tout exécuter` (≈ 5–8 min)

### Option 2 — Environnement local
```bash
# 1. Cloner le repo
git clone git@github.com:Constantin8585/Data-Projet-tourpedia-paris-analysis.git
cd tourpedia-paris-analysis

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer Jupyter
jupyter notebook TourPedia_Paris_Analysis.ipynb
```

---

## Stack technique

| Outil | Usage |
|---|---|
| **Pandas** | Chargement JSONL, transformation, nettoyage, agrégations |
| **NumPy** | Calculs vectorisés (scores composites, normalisation) |
| **Scikit-learn** | K-Means, Random Forest, Gradient Boosting, Ridge, Pipeline, CV |
| **Plotly** | Graphiques interactifs, scatter map, dashboard |
| **Folium** | Cartes interactives (HeatMap, MarkerCluster) |
| **Matplotlib / Seaborn** | Visualisations statiques haute résolution |

---

## Description du dataset

| Champ | Type | Description |
|---|---|---|
| `_id` | int | Identifiant unique du lieu |
| `name` | str | Nom du lieu |
| `category` | str | `restaurant` / `poi` / `attraction` / `accommodation` |
| `location.coord` | GeoJSON | Coordonnées GPS (longitude, latitude) |
| `location.address` | str | Adresse postale |
| `reviews` | list | Liste d'avis (source, texte, polarity 0-10, rating 0-5) |
| `nbReviews` | int | Nombre total d'avis déclarés |

**Source** : [TourPedia Open Dataset](http://tour-pedia.org) — données collectées via les APIs
Foursquare, Google Places et Facebook.

---

## Pistes d'amélioration

- [ ] **NLP** : analyse de sentiment des textes avec CamemBERT (Hugging Face)
- [ ] **Séries temporelles** : évolution des avis 2011–2015
- [ ] **API de recommandation** : déploiement du modèle via FastAPI
- [ ] **MLOps** : tracking des expériences avec MLflow
- [ ] **Conteneurisation** : Dockerfile pour déploiement reproductible

---

## Auteur

**Constantin GADEGBE**
Etudiant 
[gadegbeabednego@gmail.com](mailto:gadegbeabednego@gmail.com) · [LinkedIn](https://www.linkedin.com/in/constantin-gadegbe)
