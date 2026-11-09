"""DBSCAN pour la segmentation de clients (Mall Customers).

Les données sont standardisées avant DBSCAN : sans cela, eps n'a pas
le même sens pour le revenu (k$) et pour le score de dépenses (1-100).
Le graphe des k-distances aide à choisir eps (on prend la valeur au « coude »).
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from utils import MALL_CSV, load_mall_customers, save_figure


def k_distance_plot(X_scaled, min_samples):
    """Trace la distance au min_samples-ième voisin, triée, pour choisir eps."""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.sort(distances[:, -1]))
    ax.set_xlabel("Points triés")
    ax.set_ylabel(f"Distance au {min_samples}e voisin")
    ax.set_title("Choix de eps : graphe des k-distances")
    save_figure(fig, "mall_dbscan_k_distance.png")


def main(eps, min_samples, data_path):
    X, cols = load_mall_customers(data_path)
    X_scaled = StandardScaler().fit_transform(X)

    k_distance_plot(X_scaled, min_samples)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"eps = {eps}, min_samples = {min_samples}")
    print(f"Clusters trouvés : {n_clusters} | points de bruit : {n_noise}/{len(X)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    core = labels != -1
    ax.scatter(X[core, 0], X[core, 1], c=labels[core], cmap="viridis", s=25)
    ax.scatter(X[~core, 0], X[~core, 1], c="lightgray", marker="x", s=25, label="Bruit")
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_title(f"Segmentation clients : DBSCAN ({n_clusters} clusters)")
    ax.legend()
    save_figure(fig, "mall_dbscan.png")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, default=0.35, help="rayon du voisinage (données standardisées)")
    parser.add_argument("--min-samples", type=int, default=5, help="nombre minimum de voisins")
    parser.add_argument("--data", default=MALL_CSV, help="chemin vers Mall_Customers.csv")
    args = parser.parse_args()
    main(args.eps, args.min_samples, args.data)
