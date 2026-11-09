"""Visualisation du dataset Iris avant la classification."""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from utils import save_figure


def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(X[:, 2], X[:, 3], c=y, cmap="viridis", s=25)
    ax.set_xlabel(iris.feature_names[2])   # colonne 2 = longueur du pétale
    ax.set_ylabel(iris.feature_names[3])   # colonne 3 = largeur du pétale
    ax.set_title("Iris : les 3 espèces selon les pétales")
    ax.legend(scatter.legend_elements()[0], iris.target_names, title="Espèce")
    save_figure(fig, "clf_iris_explore.png")
    plt.show()


if __name__ == "__main__":
    main()
