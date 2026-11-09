"""SVR (Support Vector Regression) avec noyau RBF.

Le SVR est sensible à l'échelle des variables : la standardisation est indispensable.
L'entraînement peut prendre environ une minute (15 000 exemples).
"""
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils import evaluate, load_data, plot_predictions


def build_model():
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1))


def main():
    X_train, X_test, y_train, y_test, _ = load_data()
    model = build_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred, "SVR (noyau RBF)")
    plot_predictions(y_test, y_pred, "SVR (noyau RBF)", "reg_svr.png")
    plt.show()


if __name__ == "__main__":
    main()
