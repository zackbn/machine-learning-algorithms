"""KNN (k plus proches voisins) sur Iris.

Le nombre de voisins k est choisi par validation croisée sur le train,
jamais sur le test.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import evaluate, load_data, save_figure


def build_model(k=6):
    # p=1 : distance de Manhattan. Le KNN repose sur des distances,
    # donc les variables sont standardisées.
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, p=1))


def main():
    X_train, X_test, y_train, y_test, classes = load_data("iris")

    ks = range(1, 21)
    cv_scores = [cross_val_score(build_model(k), X_train, y_train, cv=5).mean() for k in ks]
    best_k = ks[int(np.argmax(cv_scores))]
    print(f"Meilleur k (validation croisée 5 plis) : {best_k}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(ks), cv_scores, marker="o")
    ax.axvline(best_k, color="red", linestyle="--", label=f"k = {best_k}")
    ax.set_xlabel("Nombre de voisins k")
    ax.set_ylabel("Accuracy moyenne (validation croisée)")
    ax.set_title("KNN : choix de k")
    ax.legend()
    save_figure(fig, "clf_knn_choose_k.png")

    model = build_model(best_k).fit(X_train, y_train)
    evaluate(model, X_test, y_test, classes, f"KNN (k = {best_k})", "clf_knn_confusion.png")

    # Prédiction pour une nouvelle fleur
    new_flower = np.array([[5.6, 3.4, 1.4, 0.1]])
    print(f"Nouvelle fleur {new_flower[0].tolist()} -> {classes[model.predict(new_flower)[0]]}")
    plt.show()


if __name__ == "__main__":
    main()
