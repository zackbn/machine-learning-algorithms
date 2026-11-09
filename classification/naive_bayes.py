"""Naive Bayes gaussien sur Iris."""
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

from utils import evaluate, load_data


def build_model():
    return GaussianNB()


def main():
    X_train, X_test, y_train, y_test, classes = load_data("iris")
    model = build_model().fit(X_train, y_train)
    evaluate(model, X_test, y_test, classes, "Naive Bayes gaussien", "clf_nb_confusion.png")
    plt.show()


if __name__ == "__main__":
    main()
