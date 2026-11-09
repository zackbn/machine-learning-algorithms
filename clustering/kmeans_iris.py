"""K-Means sur le dataset Iris.

- Choix du nombre de clusters avec la méthode du coude (elbow method)
- Clustering K-Means avec k = 3
- Comparaison avec les vraies espèces (Adjusted Rand Index)
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, silhouette_score

from utils import save_figure

RANDOM_STATE = 42


def elbow_method(X, k_max=9):
    """Calcule l'inertie de K-Means pour k = 1..k_max."""
    ks = range(1, k_max + 1)
    inertias = [
        KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X).inertia_
        for k in ks
    ]
    return list(ks), inertias


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    # 1. Méthode du coude
    ks, inertias = elbow_method(X)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("Nombre de clusters k")
    ax.set_ylabel("Inertie (somme des distances au carré)")
    ax.set_title("Iris : méthode du coude")
    save_figure(fig, "iris_elbow.png")

    # 2. K-Means avec k = 3
    km = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    print(f"Silhouette score       : {silhouette_score(X, labels):.3f}")
    print(f"Adjusted Rand Index    : {adjusted_rand_score(y, labels):.3f}")

    # Affichage sur les pétales (colonnes 2 et 3)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 2], X[:, 3], c=labels, cmap="viridis", s=25)
    ax.scatter(centers[:, 2], centers[:, 3], marker="*", color="red", s=200, label="Centres")
    ax.set_xlabel(iris.feature_names[2])
    ax.set_ylabel(iris.feature_names[3])
    ax.set_title("Iris : K-Means (k = 3)")
    ax.legend()
    save_figure(fig, "iris_kmeans.png")

    plt.show()


if __name__ == "__main__":
    main()
