"""Compare tous les modèles de régression sur le même découpage train / test.

Écrit le tableau des résultats dans results/regression_results.md.
"""
import time

import matplotlib.pyplot as plt

import linear_regression
import polynomial_regression
import random_forest_regression
import svr_regression
from utils import IMAGES_DIR, ROOT, evaluate, load_data

MODELS = {
    "Régression linéaire": linear_regression.build_model,
    "Polynomiale (degré 2)": polynomial_regression.build_model,
    "Random Forest": random_forest_regression.build_model,
    "SVR (RBF)": svr_regression.build_model,
}


def main():
    X_train, X_test, y_train, y_test, _ = load_data()
    rows = []
    for name, build in MODELS.items():
        start = time.perf_counter()
        model = build().fit(X_train, y_train)
        duration = time.perf_counter() - start
        scores = evaluate(y_test, model.predict(X_test), name)
        rows.append((name, scores, duration))

    rows.sort(key=lambda r: -r[1]["R²"])
    lines = [
        "| Modèle | R² | RMSE | MAE | Temps d'entraînement (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, s, t in rows:
        lines.append(f"| {name} | {s['R²']:.3f} | {s['RMSE']:.3f} | {s['MAE']:.3f} | {t:.1f} |")
    table = "\n".join(lines)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "regression_results.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh([r[0] for r in rows][::-1], [r[1]["R²"] for r in rows][::-1])
    ax.set_xlabel("R² sur le jeu de test")
    ax.set_xlim(0, 1)
    ax.set_title("Comparaison des modèles de régression")
    IMAGES_DIR.mkdir(exist_ok=True)
    fig.savefig(IMAGES_DIR / "reg_comparison.png", dpi=120, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
