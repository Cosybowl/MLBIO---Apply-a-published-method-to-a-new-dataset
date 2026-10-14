MARINO Raphaël

# Application de la méthode de Golub et al. au jeu de données WDBC

## Linux et WSL installation uv

Dans un terminal Linux/WSL, installez uv si vous ne l'avez pas déjà:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Exécuter le projet

Depuis la racine du projet, pour lancer le projet :

```bash
uv sync
uv run mlbio
```

## Sources et références

- Le sujet : Assignment 2: Apply a published method to a new dataset
- Golub, T. R. et al. (1999). Molecular classification of cancer: class discovery and class prediction by gene expression monitoring : https://www.science.org/doi/10.1126/science.286.5439.531
- Wolberg, W., Mangasarian, O., Street, N. et Street, W. Breast Cancer Wisconsin (Diagnostic) : https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- RepeatedStratifiedKFold : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html
- LogisticRegression : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- StandardScaler : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- métriques ROC : https://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc
- Matrice de confusion : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html