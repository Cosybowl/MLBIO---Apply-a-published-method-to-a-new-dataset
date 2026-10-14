import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from .golub_weighted import golub_weighted_voting
from .data_exploration import load_data, data_exploration, resultats_graph


# Res du calcul du personal seed
seed = 24
print(f"\nMa seed : {seed}\n")


# Calcul de moyenne de l'indice de Jaccard entre les sous-ensembles de caractéristiques
# On moyenne sur toutes les paires de folds pour mesurer la stabilité de la sélection de features
def jaccard_moyenne(features):
    res = []
    n = len(features)

    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = features[i], features[j]
            # Vaut 1 si les deux ensembles de features sont identiques, 0 s'ils sont disjoints
            jacky = len(s1.intersection(s2)) / len(s1.union(s2))
            res.append(jacky)

    return np.mean(res)


# Processus entier : exploration, entraînement, évaluation et stabilité, va appeler les autres fonctions.
def processus_entier():
    X, y = load_data()
    features = X.columns.tolist()

    print("Exploration des données et visualisation avec matplotlib")
    data_exploration(X, y)

    kfold = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = seed)

    golub_res = []
    reg_res = []

    sets_feat = []
    nb_feat = {feat: 0 for feat in features}

    first_split = True
    ress = []

    # Boucle sur les splits pour l'entraînement et l'évaluation sur les memes données comme ça équitable pour les deux méthodes
    for _, (train_ind, val_ind) in enumerate(kfold.split(X, y)):
        X_train, X_val = X.iloc[train_ind], X.iloc[val_ind]
        y_train, y_val = y.iloc[train_ind], y.iloc[val_ind]

        # Weighted voting
        y_pred_golub, scores_golub, selected_features = golub_weighted_voting(X_train, y_train, X_val, top_k = 10)

        acc_g = accuracy_score(y_val, y_pred_golub)
        bal_acc_g = balanced_accuracy_score(y_val, y_pred_golub)
        auc_g = roc_auc_score(y_val, scores_golub)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_golub).ravel()
        # La sensibilité (la part des tumeurs malignes qu’on repère) et la spécificité (la part des bénignes bien reconnues) sont calculées à partir de la matrice de confusion
        sensi_g = tp / (tp + fn)
        specifi_g = tn / (tn + fp)

        golub_res.append({"acc": acc_g, "bal_acc": bal_acc_g, "auc": auc_g, "sensibilite": sensi_g, "specificite": specifi_g})

        # Suivi des caractéristiques sélectionnées, on convertit les indices en noms de features pour plus de lisibilité
        selec_feats = [features[i] for i in selected_features]
        sets_feat.append(set(selec_feats))
        for f in selec_feats:
            nb_feat[f] += 1

        # Régression logistique et standardisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        reg = LogisticRegression(solver = "liblinear", C = 1.0, max_iter = 1000, random_state = seed)
        reg.fit(X_train_scaled, y_train)

        # Prédiction et évaluation de la régression logistique
        y_pred_rl = reg.predict(X_val_scaled)
        y_prob_rl = reg.predict_proba(X_val_scaled)[:, 1]

        acc_rl = accuracy_score(y_val, y_pred_rl)
        bal_acc_rl = balanced_accuracy_score(y_val, y_pred_rl)
        auc_rl = roc_auc_score(y_val, y_prob_rl)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_rl).ravel()
        sensi_rl = tp / (tp + fn)
        specifi_rl = tn / (tn + fp)

        reg_res.append({"acc": acc_rl, "bal_acc": bal_acc_rl, "auc": auc_rl, "sens": sensi_rl, "spec": specifi_rl})

        ress.append({"y_val": y_val.values, "pred_golub": y_pred_golub, "score_golub": scores_golub, "pred_rl": y_pred_rl, "prob_rl": y_prob_rl})

        # Affichage d'un exemple sur le premier split
        if first_split:
            print("\n\nExemple de résultats sur le premier split :")
            print(f"\nGolub - AUC : {auc_g: .4f}, Régression Logistique - AUC : {auc_rl: .4f}")
            print(f"\nGolub - Balanced Acc : {bal_acc_g: .4f}, Régression Logistique - Balanced Acc : {bal_acc_rl: .4f}")
            print(f"\nGolub - Sensibilité : {sensi_g: .4f}, Régression Logistique - Sensibilité : {sensi_rl: .4f}")
            print(f"\nGolub - Spécificité : {specifi_g: .4f}, Régression Logistique - Spécificité : {specifi_rl: .4f}")
            first_split = False

    # Calcul des moyennes et affichage des résultats
    df_g = pd.DataFrame(golub_res)
    df_rl = pd.DataFrame(reg_res)

    print("\n\nRésultats moyens sur les 15 splits :")
    print("\nGolub Weighted Voting :")
    print(df_g.mean().to_string())
    print("\nRégression Logistique :")
    print(df_rl.mean().to_string())

    # Stabilité de la sélection de caractéristiques, on affiche la fréquence de sélection de chaque feature sur les 15 splits, les features jamais choisies ne sont pas affichés
    print("\n\nStabilité de la sélection de caractéristiques :")
    print("\nFréquence de sélection sur les 15 splits :")
    s = sorted(nb_feat.items(), key = lambda x: x[1], reverse = True)
    for feat, count in s:
        if count > 0:
            print(f"  {feat}: {count}/15 ({count / 15*100:.1f}%)")

    # Indice de Jaccard moyen pour mesurer la stabilité de la sélection de features
    moy_jaccard = jaccard_moyenne(sets_feat)
    print(f"\nIndice de Jaccard moyen entre les 15 sous-ensembles : {moy_jaccard: .4f}")

    # Courbes ROC des 15 folds + moyenne, pour les deux méthodes
    resultats_graph(ress)


def main():
    processus_entier()


if __name__ == "__main__":
    main()