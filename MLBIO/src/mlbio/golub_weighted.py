import numpy as np


# Implémentation du vote pondéré de Golub
def golub_weighted_voting(X_train, y_train, X_val, top_k = 10):
    # Conversion en tableaux numpy : pour indexer avec des booleens
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)

    # Séparation des échantillons d'entraînement par classe (maligne = classe positive = 1, bénigne = 0)
    X_maligne = X_train[y_train == 1]
    X_benigne = X_train[y_train == 0]

    # Moyenne et écart-type de chaque feature, par classe (calcul vectorisé : axis = 0 donne un vecteur de taille 30 donc une valeur par feature)
    moy_maligne = X_maligne.mean(axis = 0)
    moy_benigne = X_benigne.mean(axis = 0)
    std_maligne = X_maligne.std(axis = 0)
    std_benigne = X_benigne.std(axis = 0)

    # Calcul des poids de Golub pour chaque feature
    weights = (moy_maligne - moy_benigne) / (std_maligne + std_benigne)

    # Les tops features, les midpoints et le score de chaque échantillon de validation
    features = np.argsort(np.abs(weights))[::-1][:top_k]
    midpoints = (moy_maligne[features] + moy_benigne[features]) / 2
    score = np.sum(weights[features] * (X_val[:, features] - midpoints), axis = 1)

    # Règle de décision du sujet, si le score est positif, on prédit la classe bénigne si 0 et maligne si 1
    predi = (score >= 0).astype(int)

    return predi, score, features