"""Random Forest pour la régression, avec importance des variables."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from utils import IMAGES_DIR, RANDOM_STATE, evaluate, load_data, plot_predictions


def build_model():
    # Les arbres n'ont pas besoin de données mises à l'échelle.
    return RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)


def main():
    X_train, X_test, y_train, y_test, features = load_data()
    model = build_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred, "Random Forest")

    order = np.argsort(model.feature_importances_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(features)[order], model.feature_importances_[order])
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest : importance des variables")
    IMAGES_DIR.mkdir(exist_ok=True)
    fig.savefig(IMAGES_DIR / "reg_rf_importance.png", dpi=120, bbox_inches="tight")

    plot_predictions(y_test, y_pred, "Random Forest", "reg_random_forest.png")
    plt.show()


if __name__ == "__main__":
    main()
