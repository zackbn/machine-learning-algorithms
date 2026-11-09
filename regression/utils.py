"""Chargement des données, évaluation et figures pour la régression.

Dataset : California Housing (scikit-learn).
Cible : valeur médiane des maisons d'un quartier, en centaines de milliers de dollars.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as skm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import metrics

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
RANDOM_STATE = 42


def load_data(test_size=0.25):
    """Renvoie X_train, X_test, y_train, y_test, noms des variables."""
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, list(data.feature_names)


def evaluate(y_true, y_pred, name=""):
    """Calcule MAE, MSE, RMSE, MAPE, R² avec nos fonctions et vérifie avec scikit-learn."""
    scores = {
        "MAE": metrics.mean_absolute_error(y_true, y_pred),
        "MSE": metrics.mean_squared_error(y_true, y_pred),
        "RMSE": metrics.root_mean_squared_error(y_true, y_pred),
        "MAPE (%)": metrics.mean_absolute_percentage_error(y_true, y_pred),
        "R²": metrics.r2_score(y_true, y_pred),
    }
    # Vérification : nos implémentations doivent donner le même résultat que scikit-learn
    assert np.isclose(scores["MAE"], skm.mean_absolute_error(y_true, y_pred))
    assert np.isclose(scores["MSE"], skm.mean_squared_error(y_true, y_pred))
    assert np.isclose(scores["R²"], skm.r2_score(y_true, y_pred))

    if name:
        print(f"\n=== {name} ===")
    for k, v in scores.items():
        print(f"{k:>9} : {v:.3f}")
    return scores


def plot_predictions(y_true, y_pred, title, filename):
    """Nuage valeurs réelles vs prédites : plus les points sont proches de la diagonale, mieux c'est."""
    IMAGES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=5, alpha=0.3)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, color="red", linewidth=1, label="Prédiction parfaite")
    ax.set_xlabel("Valeur réelle (100 k$)")
    ax.set_ylabel("Valeur prédite (100 k$)")
    ax.set_title(title)
    ax.legend()
    fig.savefig(IMAGES_DIR / filename, dpi=120, bbox_inches="tight")
    print(f"Figure enregistrée : images/{filename}")
    return fig
