"""Arbre de décision sur Iris, avec visualisation de l'arbre."""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

from utils import RANDOM_STATE, evaluate, load_data, save_figure


def build_model():
    # max_depth=3 limite le surapprentissage et garde un arbre lisible.
    return DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)


def main():
    X_train, X_test, y_train, y_test, classes = load_data("iris")
    model = build_model().fit(X_train, y_train)
    evaluate(model, X_test, y_test, classes, "Arbre de décision", "clf_tree_confusion.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(model, feature_names=load_iris().feature_names, class_names=classes,
              filled=True, rounded=True, ax=ax)
    ax.set_title("Arbre de décision (profondeur 3)")
    save_figure(fig, "clf_tree.png")
    plt.show()


if __name__ == "__main__":
    main()
