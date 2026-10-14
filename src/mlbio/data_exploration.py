from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve


# Chargement des données et prétraitement
def load_data(path = "src/mlbio/data/wdbc.csv"):
    df = pd.read_csv(path)

    if "target" not in df.columns and "diagnosis" in df.columns:
        diagno = df["diagnosis"]
        if pd.api.types.is_string_dtype(diagno.dtype):
            labels = diagno.str.strip().str.lower()
            target = labels.map({"malignant": 1, "benign": 0})

            if target.isna().any():
                raise ValueError(f"Erreur, certaines valeurs de diagnosis ne sont pas bonnes")
            df["target"] = target

        else:
            df["target"] = diagno
        df = df.drop(columns = ["diagnosis"])

    # Supprimer les colonnes inutiles (les ids) qui peuvent etre même un biais pour le modele
    sup_cols = [col for col in ("id", "sample_id") if col in df.columns]
    if sup_cols:
        df = df.drop(columns = sup_cols)

    # X : les 30 features numériques et y : l'étiquette (1 = maligne)
    X = df.drop(columns = ["target"])
    y = df["target"]

    return X, y


# Exploration des données et visualisation avec matplotlib
def data_exploration(X, y, out = "figures"):
    out = Path(out)
    out.mkdir(parents = True, exist_ok = True)

    print(f"Nombre d'échantillons : {X.shape[0]}")
    print(f"Nombre de caractéristiques : {X.shape[1]}")
    print(f"Distribution des classes : Maligne (1) = {sum(y == 1)}, Bénigne (0) = {sum(y == 0)}")

    # Figure 1 : effectifs par classe
    plt.figure(figsize = (7, 5))
    plt.bar(["Bénigne", "Maligne"], [(y == 0).sum(), (y == 1).sum()], color = ["blue", "red"])
    plt.title("Nombre d'échantillons par diagnostic")
    plt.ylabel("Nombre d'échantillons")
    plt.tight_layout()
    plt.savefig(out / "diagnostics.png")
    plt.close()

    # Figure 2 : rayon et texture moyens selon le diagnostic, pour visualiser le lien avec la morphologie
    plt.figure(figsize = (7, 5))
    plt.scatter(X.loc[y == 0, "mean_radius"], X.loc[y == 0, "mean_texture"], label = "Bénigne", color = "blue")
    plt.scatter(X.loc[y == 1, "mean_radius"], X.loc[y == 1, "mean_texture"], label = "Maligne", color = "red")
    plt.title("Rayon moyen et texture moyenne")
    plt.xlabel("Rayon moyen")
    plt.ylabel("Texture moyenne")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "rayon_texture.png")
    plt.close()

    # Figure 3 : surface moyenne des cellules selon le diagnostic
    moy_area_b = X.loc[y == 0, "mean_area"].mean()
    moy_area_m = X.loc[y == 1, "mean_area"].mean()
    plt.figure(figsize = (7, 5))
    plt.bar(["Bénigne", "Maligne"], [moy_area_b, moy_area_m], color = ["blue", "red"])
    plt.title("Surface moyenne des cellules par diagnostic")
    plt.ylabel("Surface moyenne")
    plt.tight_layout()
    plt.savefig(out / "surface_moyenne.png")
    plt.close()

    print(f"Graphiques d'exploration enregistrés dans : {out}")


# La courbe ROC de chaque modèle est tracée pour chaque fold, et la moyenne des courbes ROC est affichée
def resultats_graph(res, out = "figures"):
    out = Path(out)
    out.mkdir(exist_ok = True)

    # Courbes ROC : un panneau par méthode, axe y partagé pour une meilleure comparaison visuelle
    fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey = True)
    grid = np.linspace(0, 1, 100)

    # ROC Golub
    # Liste des taux de true positifs pour chaque fold
    tps = []
    for e in res:
        fp, tp, _ = roc_curve(e["y_val"], e["score_golub"])
        # Tracé de la courbe ROC pour chaque fold avec transparence pour rendre la moyenne plus visible
        axes[0].plot(fp, tp, color = "tab:red", alpha = 0.2)
        tps.append(np.interp(grid, fp, tp))

    axes[0].plot(grid, np.mean(tps, axis = 0), color = "black", label = "Moyenne des folds")
    axes[0].plot([0, 1], [0, 1], color = "black")
    axes[0].set_title("ROC - Golub")
    axes[0].set_xlabel("Taux de faux positifs (FP)")
    axes[0].set_ylabel("Taux de vrais positifs (TP)")
    axes[0].legend(loc = "lower right")

    # ROC Régression Logistique
    tps = []
    for e in res:
        fp, tp, _ = roc_curve(e["y_val"], e["prob_rl"])
        axes[1].plot(fp, tp, color = "tab:blue", alpha = 0.2)
        tps.append(np.interp(grid, fp, tp))

    axes[1].plot(grid, np.mean(tps, axis = 0), color = "black", label = "Moyenne des folds")
    axes[1].plot([0, 1], [0, 1], color = "black")
    axes[1].set_title("ROC - Régression Logistique")
    axes[1].set_xlabel("Taux de faux positifs (FP)")
    axes[1].set_ylabel("Taux de vrais positifs (TP)")
    axes[1].legend(loc = "lower right")

    fig.tight_layout()
    fig.savefig(out / "roc.png")
    plt.close(fig)

    print(f"Courbes ROC de comparaison enregistrées dans : {out} (roc.png)")