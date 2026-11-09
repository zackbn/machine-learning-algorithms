"""Chargement des données, évaluation et figures pour la classification.

Datasets (inclus dans scikit-learn, aucun téléchargement) :
- Iris : 3 espèces de fleurs, 4 variables
- Breast Cancer : tumeur maligne (0) ou bénigne (1), 30 variables
"""
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
RANDOM_STATE = 42


def load_data(name="iris", test_size=0.2):
    """Renvoie X_train, X_test, y_train, y_test, noms des classes.

    stratify=y garde la même proportion de chaque classe dans le train et le test.
    """
    data = load_iris() if name == "iris" else load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size,
        random_state=RANDOM_STATE, stratify=data.target,
    )
    return X_train, X_test, y_train, y_test, list(data.target_names)


def evaluate(model, X_test, y_test, class_names, title, filename):
    """Affiche l'accuracy, le rapport de classification et enregistre la matrice de confusion."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {title} ===")
    print(f"Accuracy : {acc:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"{title}\naccuracy = {acc:.3f}")
    save_figure(fig, filename)
    return acc


def save_figure(fig, filename):
    IMAGES_DIR.mkdir(exist_ok=True)
    fig.savefig(IMAGES_DIR / filename, dpi=120, bbox_inches="tight")
    print(f"Figure enregistrée : images/{filename}")
