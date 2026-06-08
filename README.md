# Parkinson's Disease Detection via Voice Signal Analysis

This repository contains the machine learning notebooks developed for my undergraduate thesis at BRAC University. The project investigates automated Parkinson's disease diagnosis using voice signal features, exploring how different feature selection strategies and dimensionality reduction affect classification performance.

---

## Dataset
- [Kaggle Dataset 1 — https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set]
- [Kaggle Dataset 2 — https://www.kaggle.com/datasets/dipayanbiswas/parkinsons-disease-speech-signal-features]

---

## Notebooks

| Notebook | Description |
|---|---|
| `dataset2_50_feature.ipynb` | Top 50 features selected via SelectKBest with ANOVA F-score |
| `dataset2_70_feature.ipynb` | Top 70 features selected via SelectKBest with ANOVA F-score |
| `dataset2_80_feature.ipynb` | Top 80 features selected via SelectKBest with ANOVA F-score |
| `dataset2_100_feature.ipynb` | Full 100-feature set with GAN-based data augmentation |
| `pca_80_components.ipynb` | Dimensionality reduction to 80 components via PCA |

---

## Models Used
Random Forest, Extra Trees, Decision Tree, XGBoost, CatBoost, and a Hybrid Soft Voting Ensemble (RF + XGBoost + CatBoost).

---

## Explainability
SHAP (feature importance), LIME (local instance explanations), Permutation Importance, and Morris Sensitivity Analysis.

---

## Tech Stack
Python · scikit-learn · XGBoost · CatBoost · Keras · SHAP · LIME · pandas · numpy
