"""Compare les classifieurs sur Iris et Breast Cancer par validation croisée (5 plis).

Écrit le tableau des résultats dans results/classification_results.md.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import decision_tree
import knn
import logistic_regression
import naive_bayes
from utils import RANDOM_STATE, ROOT, load_data, save_figure

MODELS = {
    "KNN": knn.build_model,
    "Arbre de décision": decision_tree.build_model,
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Naive Bayes": naive_bayes.build_model,
    "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    "Régression logistique": logistic_regression.build_model,
}
DATASETS = {"Iris": "iris", "Breast Cancer": "breast_cancer"}


def main():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for ds_label, ds_name in DATASETS.items():
        X_train, _, y_train, _, _ = load_data(ds_name)
        for model_name, build in MODELS.items():
            scores = cross_val_score(build(), X_train, y_train, cv=cv)
            results[(model_name, ds_label)] = (scores.mean(), scores.std())
            print(f"{ds_label:>13} | {model_name:<22} | {scores.mean():.3f} ± {scores.std():.3f}")

    header = "| Modèle | " + " | ".join(f"{d} (accuracy)" for d in DATASETS) + " |"
    lines = [header, "| --- |" + " --- |" * len(DATASETS)]
    for m in MODELS:
        cells = [f"{results[(m, d)][0]:.3f} ± {results[(m, d)][1]:.3f}" for d in DATASETS]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "classification_results.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(MODELS))
    h = 0.38
    for i, d in enumerate(DATASETS):
        means = [results[(m, d)][0] for m in MODELS]
        stds = [results[(m, d)][1] for m in MODELS]
        ax.barh(y + (i - 0.5) * h, means, height=h, xerr=stds, label=d, capsize=3)
    ax.set_yticks(y, list(MODELS))
    ax.invert_yaxis()
    ax.set_xlim(0.8, 1.0)
    ax.set_xlabel("Accuracy moyenne (validation croisée 5 plis)")
    ax.set_title("Comparaison des classifieurs")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    save_figure(fig, "clf_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
