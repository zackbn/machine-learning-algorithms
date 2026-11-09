"""K-Means pour la segmentation de clients (Mall Customers).

Variables utilisées : revenu annuel (k$) et score de dépenses (1-100).
Le nombre de clusters est choisi avec la méthode du coude.
"""
import argparse

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import MALL_CSV, load_mall_customers, save_figure

RANDOM_STATE = 42


def main(k: int, data_path):
    X, cols = load_mall_customers(data_path)

    # 1. Méthode du coude
    ks = range(1, 11)
    inertias = [
        KMeans(n_clusters=i, n_init=10, random_state=RANDOM_STATE).fit(X).inertia_
        for i in ks
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(ks), inertias, marker="o")
    ax.set_xlabel("Nombre de clusters k")
    ax.set_ylabel("Inertie")
    ax.set_title("Mall Customers : méthode du coude")
    save_figure(fig, "mall_elbow.png")

    # 2. K-Means avec le k choisi
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    print(f"k = {k} | silhouette score : {silhouette_score(X, labels):.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=25)
    ax.scatter(centers[:, 0], centers[:, 1], marker="*", color="red", s=200, label="Centres")
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_title(f"Segmentation clients : K-Means (k = {k})")
    ax.legend()
    save_figure(fig, "mall_kmeans.png")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="nombre de clusters (défaut : 5)")
    parser.add_argument("--data", default=MALL_CSV, help="chemin vers Mall_Customers.csv")
    args = parser.parse_args()
    main(args.k, args.data)
