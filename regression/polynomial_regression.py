"""Régression polynomiale : comparaison des degrés 1, 2 et 3.

Montre le compromis biais / variance : un degré plus élevé s'ajuste mieux
au train, mais peut moins bien généraliser sur le test.
"""
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from utils import IMAGES_DIR, evaluate, load_data, plot_predictions

BEST_DEGREE = 2


def build_model(degree=BEST_DEGREE):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )


def main():
    X_train, X_test, y_train, y_test, _ = load_data()

    degrees = [1, 2, 3]
    r2_train, r2_test = [], []
    for d in degrees:
        model = build_model(d).fit(X_train, y_train)
        r2_train.append(r2_score(y_train, model.predict(X_train)))
        r2_test.append(r2_score(y_test, model.predict(X_test)))
        print(f"Degré {d} | R² train = {r2_train[-1]:.3f} | R² test = {r2_test[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(degrees, r2_train, marker="o", label="Train")
    ax.plot(degrees, r2_test, marker="o", label="Test")
    ax.set_xticks(degrees)
    ax.set_xlabel("Degré du polynôme")
    ax.set_ylabel("R²")
    ax.set_title("Régression polynomiale : choix du degré")
    ax.legend()
    IMAGES_DIR.mkdir(exist_ok=True)
    fig.savefig(IMAGES_DIR / "reg_polynomial_degree.png", dpi=120, bbox_inches="tight")

    model = build_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred, f"Régression polynomiale (degré {BEST_DEGREE})")
    plot_predictions(y_test, y_pred, f"Régression polynomiale (degré {BEST_DEGREE})", "reg_polynomial.png")
    plt.show()


if __name__ == "__main__":
    main()
