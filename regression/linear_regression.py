"""Régression linéaire multiple sur California Housing (8 variables)."""
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import evaluate, load_data, plot_predictions


def build_model():
    # Le scaler est appris sur le train uniquement (dans le pipeline),
    # puis appliqué au test : pas de fuite de données.
    return make_pipeline(StandardScaler(), LinearRegression())


def main():
    X_train, X_test, y_train, y_test, features = load_data()
    model = build_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred, "Régression linéaire multiple")

    coefs = model.named_steps["linearregression"].coef_
    print("\nCoefficients (variables standardisées) :")
    for name, c in sorted(zip(features, coefs), key=lambda t: -abs(t[1])):
        print(f"  {name:>10} : {c:+.3f}")

    plot_predictions(y_test, y_pred, "Régression linéaire multiple", "reg_linear.png")
    plt.show()


if __name__ == "__main__":
    main()
